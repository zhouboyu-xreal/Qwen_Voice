const assert = require('node:assert/strict')
const { resolve } = require('node:path')
const { pathToFileURL } = require('node:url')

// Use the real Web Audio graph and preload with synthetic audio. This smoke
// never asks for microphone/screen access or touches user recordings/config.
module.exports = async function ambientAudioSmoke({ BrowserWindow, ipcMain }) {
  const chunks = []
  const captures = []
  let nativePhase = false
  ipcMain.handle('qwen-audio-agent:ambient-recording-prepare', () => ({
    sources: ['microphone', 'system'], systemCapture: nativePhase ? 'native' : 'display', captureToken: 'smoke-token',
  }))
  ipcMain.handle('qwen-audio-agent:ambient-recording-start', () => ({ recordingId: 'smoke-session' }))
  ipcMain.handle('qwen-audio-agent:ambient-recording-stop', () => { captures.push('stop'); return { state: 'finalizing' } })
  ipcMain.handle('qwen-audio-agent:ambient-native-audio-start', (_event, token) => {
    assert.equal(token, 'smoke-token')
    captures.push('native-start')
    return { backend: 'native', sampleRate: 48000 }
  })
  ipcMain.handle('qwen-audio-agent:ambient-native-audio-stop', (_event, token) => {
    assert.equal(token, 'smoke-token')
    captures.push('native-stop')
  })
  const listener = (_event, payload) => { chunks.push(payload); captures.push(payload.source) }
  ipcMain.on('qwen-audio-agent:ambient-recording-audio', listener)
  const window = new BrowserWindow({
    width: 320, height: 180, show: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true,
      backgroundThrottling: false, preload: resolve(__dirname, '../src/preload.cjs') },
  })
  try {
    await window.loadFile(resolve(__dirname, '../src/orb-unavailable.html'))
    const moduleUrl = pathToFileURL(resolve(__dirname, '../../web/src/desktop/ambient-recording-controller.js')).href
    const result = await window.webContents.executeJavaScript(`(async () => {
      const { createAmbientRecordingController } = await import(${JSON.stringify(moduleUrl)})
      const { createAmbientSystemAudioCapture } = await import(${JSON.stringify(pathToFileURL(resolve(__dirname, '../../web/src/desktop/ambient-system-audio.js')).href)})
      const generator = new AudioContext({ sampleRate: 48000 })
      await generator.resume()
      const oscillator = generator.createOscillator()
      const destination = generator.createMediaStreamDestination()
      oscillator.frequency.value = 440
      oscillator.connect(destination)
      oscillator.start()
      let localClock = null
      const controller = createAmbientRecordingController({
        desktop: window.qwenAudioAgentDesktop,
        flushMicrophone: () => controller.appendMicrophone('AQACAA==', 16000, new Date().toISOString()),
        onError: message => { throw new Error(message) },
        createSystemCapture: opts => createAmbientSystemAudioCapture({
          ...opts, mediaDevices: { getDisplayMedia: async () => destination.stream },
          createContext: () => { localClock = new AudioContext({ sampleRate: 48000 }); return localClock },
        }),
      })
      try {
        await controller.start()
        controller.appendMicrophone('AQACAA==', 16000, new Date().toISOString())
        await new Promise(resolve => setTimeout(resolve, 1300))
        await controller.stop()
        return { stopped: destination.stream.getTracks().every(track => track.readyState === 'ended'), contextClosed: localClock.state === 'closed' }
      } finally {
        controller.release()
        oscillator.stop()
        await generator.close()
      }
    })()`, true)
    assert.equal(result.stopped, true)
    assert.equal(result.contextClosed, true)
    const system = chunks.filter(chunk => chunk.source === 'system')
    assert.ok(system.length >= 1, 'real Web Audio processor must emit system PCM')
    assert.ok(system.some(chunk => Buffer.from(chunk.audio, 'base64').some(byte => byte !== 0)))
    assert.equal(system[0].sampleRate, 48000)
    assert.ok(chunks.every(chunk => chunk.recordingId === 'smoke-session' && !Number.isNaN(Date.parse(chunk.startedAt))))
    assert.equal(captures.at(-1), 'stop', 'seal must follow both track tails through preload IPC')
    nativePhase = true
    const firstNativeChunk = chunks.length
    const nativeResult = await window.webContents.executeJavaScript(`(async () => {
      const { createAmbientRecordingController } = await import(${JSON.stringify(moduleUrl)})
      let displayCalls = 0
      const original = navigator.mediaDevices.getDisplayMedia
      navigator.mediaDevices.getDisplayMedia = () => { displayCalls++; throw new Error('Native path opened screen capture') }
      const controller = createAmbientRecordingController({
        desktop: window.qwenAudioAgentDesktop,
        flushMicrophone: () => controller.appendMicrophone('AQACAA==', 16000, new Date().toISOString()),
        onError: message => { throw new Error(message) },
      })
      try {
        await controller.start()
        await controller.stop()
        return { displayCalls }
      } finally {
        controller.release()
        navigator.mediaDevices.getDisplayMedia = original
      }
    })()`, true)
    assert.equal(nativeResult.displayCalls, 0)
    assert.ok(captures.includes('native-start'))
    assert.ok(captures.includes('native-stop'))
    assert.ok(chunks.slice(firstNativeChunk).every(chunk => chunk.source === 'microphone'), 'native system PCM must bypass the renderer')
    assert.equal(captures.at(-1), 'stop')

  } finally {
    window.destroy()
    ipcMain.removeListener('qwen-audio-agent:ambient-recording-audio', listener)
    for (const method of ['start', 'stop']) ipcMain.removeHandler(`qwen-audio-agent:ambient-native-audio-${method}`)
    for (const method of ['prepare', 'start', 'stop']) ipcMain.removeHandler(`qwen-audio-agent:ambient-recording-${method}`)
  }
}
