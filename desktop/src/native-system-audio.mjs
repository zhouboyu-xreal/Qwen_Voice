import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { resolve } from 'node:path'

export function nativeSystemAudioExecutable({ packaged, resourcesPath, sourceRoot, arch = process.arch }) {
  return packaged
    ? resolve(resourcesPath, 'native/qwen-system-audio')
    : resolve(sourceRoot, 'desktop/native/bin', arch, 'qwen-system-audio')
}

function captureError(code = '') {
  return new Error(code === 'output_device_changed'
    ? '音频输出设备已切换，请重新开始环境录音。'
    : code === 'unsupported_tap_format'
      ? '当前系统音频格式不受支持，请切换音频输出设备后重试。'
      : '系统音频采集失败，请检查“系统音频录制”权限并重试。')
}

/** Native JSONL transport. Audio is delivered directly to the local recorder,
 * never through the renderer, wake-word detector or realtime Gateway. */
export class NativeSystemAudioCapture {
  constructor({ executable, onAudio, onError = () => {}, spawnImpl = spawn,
    startTimeoutMs = 60000, stopTimeoutMs = 5000, logger = {} } = {}) {
    this.executable = executable
    this.onAudio = onAudio
    this.onError = onError
    this.spawnImpl = spawnImpl
    this.startTimeoutMs = startTimeoutMs
    this.stopTimeoutMs = stopTimeoutMs
    this.logger = logger
    this.child = null
    this.ready = false
    this.recording = false
    this.stopping = false
    this.inflight = new Set()
    this.writeError = null
    this.stopPromise = null
  }

  async start() {
    if (this.child) throw new Error('System audio helper already started')
    this.closed = new Promise(resolveClose => { this.resolveClose = resolveClose })
    const startup = new Promise((resolveReady, rejectReady) => {
      this.resolveReady = resolveReady
      this.rejectReady = rejectReady
    })
    this.startupTimer = setTimeout(() => {
      this.fail(captureError('startup_timeout'))
      void this.stop().catch(() => {})
    }, this.startTimeoutMs)
    try {
      this.child = this.spawnImpl(this.executable, [], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { PATH: process.env.PATH, TMPDIR: process.env.TMPDIR, LANG: process.env.LANG },
      })
      createInterface({ input: this.child.stdout }).on('line', line => this.receive(line))
      // Drain diagnostics, but never forward raw helper output into the UI.
      this.child.stderr.resume()
      this.child.stdin.on('error', () => { if (!this.stopping) this.fail(captureError()) })
      this.child.once('error', error => {
        this.logger.error?.('desktop.system_audio.spawn_failed', { code: error.code })
        this.fail(captureError())
        this.resolveClose()
      })
      this.child.once('close', (code, signal) => {
        clearTimeout(this.startupTimer)
        if (!this.stopping) {
          this.logger.warn?.('desktop.system_audio.exited', { code, signal })
          this.fail(captureError())
        }
        this.resolveClose()
      })
    } catch (error) {
      clearTimeout(this.startupTimer)
      this.resolveClose()
      this.fail(error)
    }
    return startup
  }

  fail(error) {
    clearTimeout(this.startupTimer)
    if (!this.ready) this.rejectReady?.(error)
    else if (!this.stopping && !this.writeError) this.onError(error)
    this.writeError ||= error
  }

  receive(line) {
    if (line.length > 1024 * 1024) return this.fail(captureError('protocol'))
    let event
    try { event = JSON.parse(line) } catch { return this.fail(captureError('protocol')) }
    if (event.type === 'ready' && !this.stopping && !this.ready) {
      if (!Number.isInteger(event.sampleRate) || event.sampleRate < 8000 || event.sampleRate > 192000) {
        return this.fail(captureError('protocol'))
      }
      clearTimeout(this.startupTimer)
      this.ready = true
      this.resolveReady({ backend: 'native', sampleRate: event.sampleRate })
      return
    }
    if (event.type === 'error') {
      this.logger.warn?.('desktop.system_audio.capture_failed', { code: event.code, status: event.status })
      return this.fail(captureError(event.code))
    }
    if (event.type !== 'audio' || !this.recording) return
    if (typeof event.audio !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(event.audio)
      || !Number.isInteger(event.sampleRate) || event.sampleRate < 8000 || event.sampleRate > 192000
      || typeof event.startedAt !== 'string' || !Number.isFinite(Date.parse(event.startedAt))) {
      return this.fail(captureError('protocol'))
    }
    const audio = Buffer.from(event.audio, 'base64')
    if (!audio.length || audio.length % 2) return this.fail(captureError('protocol'))
    const operation = Promise.resolve().then(() => this.onAudio(audio, event.sampleRate, event.startedAt))
    this.inflight.add(operation)
    void operation.then(() => this.inflight.delete(operation), error => {
      this.inflight.delete(operation)
      this.fail(error)
    })
  }

  begin() {
    if (!this.ready || this.stopping || this.writeError) throw captureError()
    this.recording = true
    this.child.stdin.write('begin\n')
  }

  stop() {
    if (this.stopPromise) return this.stopPromise
    this.stopping = true
    clearTimeout(this.startupTimer)
    this.rejectReady?.(new Error('系统音频采集已取消'))
    this.stopPromise = (async () => {
      if (this.child) {
        this.child.stdin.end('stop\n')
        let timer
        const ended = await Promise.race([
          this.closed.then(() => true),
          new Promise(resolveTimeout => { timer = setTimeout(() => resolveTimeout(false), this.stopTimeoutMs) }),
        ])
        clearTimeout(timer)
        if (!ended) { this.child.kill('SIGKILL'); throw captureError('stop_timeout') }
      }
      // Child 'close' follows stdout EOF. Every last PCM frame has now been
      // parsed; wait for the recorder to persist it before sealing its WAV.
      await Promise.allSettled([...this.inflight])
      this.recording = false
      this.ready = false
      this.child = null
      if (this.writeError) throw this.writeError
    })()
    return this.stopPromise
  }
}
