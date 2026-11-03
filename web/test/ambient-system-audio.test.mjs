import assert from 'node:assert/strict'
import test from 'node:test'
import { createAmbientSystemAudioCapture } from '../src/desktop/ambient-system-audio.js'
import { createAmbientRecordingController } from '../src/desktop/ambient-recording-controller.js'

function graph({ audioState = 'live', getDisplayMediaError } = {}) {
  class Track extends EventTarget {
    readyState = 'live'
    stopped = false
    stop() { this.readyState = 'ended'; this.stopped = true }
  }
  const audio = new Track()
  audio.readyState = audioState
  const video = new Track()
  const stream = { getTracks: () => [audio, video], getAudioTracks: () => [audio] }
  const node = () => ({ connect() {}, disconnect() {} })
  const processor = node()
  const sink = { ...node(), gain: { value: 1 } }
  const context = {
    sampleRate: 8000, resume: async () => {}, close: async () => { context.closed = true },
    createMediaStreamSource: () => node(), createScriptProcessor: () => processor,
    createGain: () => sink, destination: {},
  }
  const calls = []
  const ended = []
  let constraints
  const capture = createAmbientSystemAudioCapture({
    mediaDevices: { getDisplayMedia: async value => {
      constraints = value
      if (getDisplayMediaError) throw getDisplayMediaError
      return stream
    } },
    createContext: () => context,
    onAudio: (...args) => calls.push(args), onEnded: message => ended.push(message),
    now: () => Date.parse('2026-11-03T01:00:01Z'),
  })
  const frame = (samples = 2000) => processor.onaudioprocess?.({
    inputBuffer: { getChannelData: () => new Float32Array(samples).fill(0.25) },
  })
  return { capture, calls, ended, audio, video, processor, context, sink, frame, constraints: () => constraints }
}

test('system graph records only after commit, preserves rate/tail, and never plays capture back', async () => {
  const g = graph()
  await g.capture.start()
  assert.equal(g.constraints().systemAudio, 'include')
  assert.equal(g.constraints().audio.echoCancellation, false)
  assert.equal(g.sink.gain.value, 0)
  g.frame()
  assert.equal(g.calls.length, 0)
  g.capture.begin()
  for (let i = 0; i < 4; i++) g.frame()
  assert.equal(g.calls.length, 1)
  assert.equal(g.calls[0][1], 8000)
  assert.equal(Buffer.from(g.calls[0][0], 'base64').length, 16000)
  assert.equal(g.calls[0][2], '2026-11-03T01:00:00.750Z')
  g.frame(123)
  g.capture.stop()
  assert.equal(Buffer.from(g.calls[1][0], 'base64').length, 246)
  assert.equal(g.audio.stopped, true)
  assert.equal(g.video.stopped, true)
  assert.equal(g.context.closed, true)
  assert.equal(g.ended.length, 0)
})

test('missing or ended audio track fails visibly and releases video capture', async () => {
  const g = graph({ audioState: 'ended' })
  await assert.rejects(g.capture.start(), /未获取到系统音频/)
  assert.equal(g.video.stopped, true)
})

test('unexpected native track end flushes audio and reports interruption once', async () => {
  const g = graph()
  await g.capture.start()
  g.capture.begin()
  g.frame()
  g.audio.dispatchEvent(new Event('ended'))
  g.video.dispatchEvent(new Event('ended'))
  assert.equal(g.calls.length, 1)
  assert.equal(g.ended.length, 1)
  assert.equal(g.processor.onaudioprocess, null)
})

function controllerFixture({ sources = ['microphone', 'system'], failCapture = false } = {}) {
  const events = []
  let callbacks
  let tick
  let currentTime = 0
  const desktop = {
    prepareAmbientRecording: async () => ({ sources }),
    startAmbientRecording: async () => { events.push('start'); return { recordingId: 'fixture-session' } },
    appendAmbientRecordingAudio: (audio, rate, options) => events.push({ audio, rate, ...options }),
    stopAmbientRecording: async options => { events.push({ stop: options }); return { state: 'finalizing' } },
  }
  let controller
  controller = createAmbientRecordingController({
    desktop,
    flushMicrophone: () => controller.appendMicrophone('mic-tail', 16000, 'time-mic'),
    onError: message => events.push({ error: message }),
    createSystemCapture: opts => {
      callbacks = opts
      return {
        start: async () => { if (failCapture) throw new Error('permission denied') },
        begin: () => events.push('begin'),
        stop: () => opts.onAudio('system-tail', 48000, 'time-system'),
      }
    },
    now: () => currentTime,
    setIntervalFn: fn => { tick = fn; return 1 }, clearIntervalFn: () => { tick = null },
  })
  return { controller, events, callbacks: () => callbacks, tick: () => tick?.(), time: value => { currentTime = value } }
}

test('controller labels both sources and flushes both tails before sealing one shared session', async () => {
  const g = controllerFixture()
  await g.controller.start()
  g.controller.appendMicrophone('mic', 16000, 'time-mic')
  g.callbacks().onAudio('system', 48000, 'time-system')
  await g.controller.stop()
  assert.deepEqual(g.events.slice(2).map(event => event.source || 'stop'), ['microphone', 'system', 'microphone', 'system', 'microphone', 'stop'])
  assert.equal(g.events[2].recordingId, 'fixture-session')
  assert.equal(g.events[3].startedAt, 'time-system')
  g.controller.appendMicrophone('late', 16000)
  assert.equal(g.events.length, 8)
})

test('microphone-only mode does not acquire system capture and permission failure never commits a recording', async () => {
  const mic = controllerFixture({ sources: ['microphone'] })
  await mic.controller.start()
  assert.equal(mic.callbacks(), undefined)
  await mic.controller.stop()
  const denied = controllerFixture({ failCapture: true })
  await assert.rejects(denied.controller.start(), /permission denied/)
  assert.equal(denied.events.includes('start'), false)
  assert.equal(denied.events.length, 1)
})

test('sample starvation stops the recording instead of reporting a healthy missing microphone track', async () => {
  const g = controllerFixture()
  await g.controller.start()
  g.time(11000)
  g.callbacks().onAudio('system', 48000, 'time-system')
  g.tick()
  await Promise.resolve()
  assert.match(g.events.find(event => event.error)?.error, /麦克风音频采集中断/)
  assert.match(g.events.at(-1).stop.captureError, /麦克风/)
})


test('native renderer adapter never opens display capture or forwards system PCM through the renderer', async () => {
  let callback
  let active = 0
  let starts = 0
  let stops = 0
  const capture = createAmbientSystemAudioCapture({
    backend: 'native', captureToken: 'current',
    mediaDevices: { getDisplayMedia: () => assert.fail('no screen sharing on native path') },
    desktop: {
      onAmbientSystemAudioProgress: fn => { callback = fn; return () => { callback = null } },
      startNativeSystemAudioCapture: async token => { assert.equal(token, 'current'); starts++ },
      stopNativeSystemAudioCapture: async token => { assert.equal(token, 'current'); stops++ },
    },
    onAudio: () => assert.fail('native PCM belongs to main-process recorder'),
    onActivity: () => { active++ },
  })
  await capture.start()
  capture.begin()
  callback({ captureToken: 'stale' })
  callback({ captureToken: 'current' })
  assert.equal(active, 1)
  assert.equal(starts, 1)
  await capture.stop()
  await capture.stop()
  assert.equal(stops, 1)
  assert.equal(callback, null)
})
