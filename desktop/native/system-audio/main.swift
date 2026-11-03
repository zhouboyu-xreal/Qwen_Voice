import Foundation
import CoreAudio
import AudioToolbox
import Darwin

// Core Audio process taps only capture outgoing audio. No ScreenCaptureKit,
// microphone device, video stream, audio playback or echo processor is used.
struct CaptureFailure: Error {
    let operation: String
    let status: OSStatus
}

func checked(_ status: OSStatus, _ operation: String) throws {
    if status != noErr { throw CaptureFailure(operation: operation, status: status) }
}

let outputQueue = DispatchQueue(label: "qwen.system-audio.protocol")
let timestampFormatter: ISO8601DateFormatter = {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter
}()

func emit(_ event: [String: Any]) {
    outputQueue.async {
        guard var data = try? JSONSerialization.data(withJSONObject: event) else { return }
        data.append(0x0a)
        try? FileHandle.standardOutput.write(contentsOf: data)
    }
}

// The tap's Float32 channels are downmixed to PCM16; no noise suppression,
// gain control, speech filtering or echo cancellation is applied.
func pcm16(_ channels: [[Float]]) -> Data {
    guard let frames = channels.map({ $0.count }).min(), frames > 0 else { return Data() }
    var result = Data(capacity: frames * 2)
    for frame in 0..<frames {
        var mixed: Float = 0
        for channel in channels { mixed += channel[frame].isFinite ? channel[frame] : 0 }
        mixed = max(-1, min(1, mixed / Float(channels.count)))
        var sample = Int16(mixed * Float(Int16.max)).littleEndian
        withUnsafeBytes(of: &sample) { result.append(contentsOf: $0) }
    }
    return result
}

@available(macOS 14.2, *)
final class SystemAudioCapture {
    private var tap: AudioObjectID = kAudioObjectUnknown
    private var aggregate: AudioObjectID = kAudioObjectUnknown
    private var io: AudioDeviceIOProcID?
    private let audioQueue = DispatchQueue(label: "qwen.system-audio.capture")
    private var recording = false
    private var stopping = false
    private var fatalScheduled = false
    private var format = AudioStreamBasicDescription()
    private var pending = Data()
    private var chunkStart: TimeInterval?
    private let epochOffset = Date().timeIntervalSince1970
        - Double(AudioConvertHostTimeToNanos(AudioGetCurrentHostTime())) / 1_000_000_000
    private var routeAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    private var routeListener: AudioObjectPropertyListenerBlock?

    func start() throws {
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.name = "Qwen Audio Agent System Audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try checked(AudioHardwareCreateProcessTap(description, &tap), "create_tap")
        var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try checked(AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &format), "tap_format")
        guard format.mFormatID == kAudioFormatLinearPCM,
              format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              format.mFormatFlags & kAudioFormatFlagIsBigEndian == 0,
              format.mBitsPerChannel == 32,
              format.mSampleRate >= 8000, format.mSampleRate <= 192000,
              format.mSampleRate.rounded() == format.mSampleRate else {
            throw CaptureFailure(operation: "unsupported_tap_format", status: kAudioHardwareUnsupportedOperationError)
        }
        let device: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Qwen Audio Agent Recording",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            // Do not wait for the first playing application before starting.
            kAudioAggregateDeviceTapAutoStartKey: false,
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapUIDKey: description.uuid.uuidString,
                kAudioSubTapDriftCompensationKey: true,
            ]],
        ]
        try checked(AudioHardwareCreateAggregateDevice(device as CFDictionary, &aggregate), "create_aggregate")
        var aliveAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsAlive,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var alive: UInt32 = 0
        for _ in 0..<60 {
            var aliveSize = UInt32(MemoryLayout<UInt32>.size)
            try checked(AudioObjectGetPropertyData(aggregate, &aliveAddress, 0, nil, &aliveSize, &alive), "device_alive")
            if alive != 0 { break }
            Thread.sleep(forTimeInterval: 0.05)
        }
        guard alive != 0 else { throw CaptureFailure(operation: "device_not_ready", status: kAudioHardwareNotRunningError) }
        try checked(AudioDeviceCreateIOProcIDWithBlock(&io, aggregate, audioQueue) {
            [weak self] _, input, inputTime, _, _ in
            self?.receive(input, at: inputTime.pointee)
        }, "create_io")
        try checked(AudioDeviceStart(aggregate, io), "start_audio")
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.fail("output_device_changed", status: noErr)
        }
        routeListener = listener
        try checked(AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
            &routeAddress, .main, listener), "watch_output_device")
        emit(["type": "ready", "sampleRate": Int(format.mSampleRate), "channels": 1, "backend": "core-audio-tap"])
    }

    func begin() {
        audioQueue.async { [self] in
            pending.removeAll(keepingCapacity: true)
            chunkStart = nil
            recording = true
        }
    }

    private func receive(_ input: UnsafePointer<AudioBufferList>, at time: AudioTimeStamp) {
        guard recording else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        var channels: [[Float]] = []
        for buffer in buffers {
            guard let bytes = buffer.mData, buffer.mNumberChannels > 0 else { continue }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            let channelCount = Int(buffer.mNumberChannels)
            let frames = count / channelCount
            let samples = bytes.assumingMemoryBound(to: Float.self)
            for channel in 0..<channelCount {
                channels.append((0..<frames).map { samples[$0 * channelCount + channel] })
            }
        }
        let data = pcm16(channels)
        guard !data.isEmpty else { return }
        let frames = data.count / 2
        let start = time.mFlags.contains(.hostTimeValid)
            ? epochOffset + Double(AudioConvertHostTimeToNanos(time.mHostTime)) / 1_000_000_000
            : Date().timeIntervalSince1970 - Double(frames) / format.mSampleRate
        if let previous = chunkStart {
            let expected = previous + Double(pending.count / 2) / format.mSampleRate
            if abs(start - expected) > 0.25 { flush(); chunkStart = start }
        } else { chunkStart = start }
        pending.append(data)
        let bytesPerSecond = Int(format.mSampleRate) * 2
        while pending.count >= bytesPerSecond {
            send(Data(pending.prefix(bytesPerSecond)), at: chunkStart!)
            pending.removeFirst(bytesPerSecond)
            chunkStart! += 1
        }
    }

    private func send(_ pcm: Data, at start: TimeInterval) {
        // Format dates on the same queue as JSON writes to avoid a shared
        // formatter crossing the audio/control threads.
        let sampleRate = Int(format.mSampleRate)
        outputQueue.async {
            let event: [String: Any] = ["type": "audio", "sampleRate": sampleRate,
                "audio": pcm.base64EncodedString(),
                "startedAt": timestampFormatter.string(from: Date(timeIntervalSince1970: start))]
            guard var data = try? JSONSerialization.data(withJSONObject: event) else { return }
            data.append(0x0a)
            try? FileHandle.standardOutput.write(contentsOf: data)
        }
    }

    private func flush() {
        if let start = chunkStart, !pending.isEmpty { send(pending, at: start) }
        pending.removeAll(keepingCapacity: true)
        chunkStart = nil
    }

    func stop() {
        guard !stopping else { return }
        stopping = true
        if let listener = routeListener {
            AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &routeAddress, .main, listener)
            routeListener = nil
        }
        if aggregate != kAudioObjectUnknown, let proc = io { AudioDeviceStop(aggregate, proc) }
        audioQueue.sync { recording = false; flush() }
        if aggregate != kAudioObjectUnknown, let proc = io { AudioDeviceDestroyIOProcID(aggregate, proc); io = nil }
        if aggregate != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(aggregate); aggregate = kAudioObjectUnknown }
        if tap != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tap); tap = kAudioObjectUnknown }
        emit(["type": "stopped"])
        outputQueue.sync {}
    }

    func fail(_ operation: String, status: OSStatus) {
        guard !fatalScheduled else { return }
        fatalScheduled = true
        emit(["type": "error", "code": operation, "status": Int(status)])
        stop()
        exit(1)
    }
}

if CommandLine.arguments.contains("--self-test") {
    let actual = pcm16([[0, 1, -1, .nan, 0.5], [0, 1, -1, 0, -0.5]])
    let expected: [Int16] = [0, Int16.max, -Int16.max, 0, 0]
    let encoded = expected.flatMap { value -> [UInt8] in
        let bits = UInt16(bitPattern: value)
        return [UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)]
    }
    guard actual == Data(encoded), pcm16([]).isEmpty else { exit(1) }
    emit(["type": "self-test-ok"])
    outputQueue.sync {}
    exit(0)
}

if #available(macOS 14.2, *) {
    let capture = SystemAudioCapture()
    do { try capture.start() }
    catch let failure as CaptureFailure { capture.fail(failure.operation, status: failure.status) }
    catch { capture.fail("startup_failed", status: kAudioHardwareUnspecifiedError) }
    var signals: [DispatchSourceSignal] = []
    for number in [SIGTERM, SIGINT] {
        signal(number, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
        source.setEventHandler { capture.stop(); exit(0) }
        source.resume()
        signals.append(source)
    }
    signal(SIGPIPE, SIG_IGN)
    DispatchQueue.global().async {
        while let command = readLine() {
            if command == "begin" { capture.begin() }
            if command == "stop" { break }
        }
        DispatchQueue.main.async { capture.stop(); exit(0) }
    }
    withExtendedLifetime(signals) { dispatchMain() }
} else {
    emit(["type": "error", "code": "unsupported_os"])
    outputQueue.sync {}
    exit(1)
}
