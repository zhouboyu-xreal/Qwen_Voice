// A real microphone/speaker experiment, deliberately separate from automated smoke tests.
// Playback runs in afplay (outside Chromium) and neither audio nor transcripts enter PWM.
const { app, BrowserWindow, ipcMain, systemPreferences } = require('electron')
const { spawn } = require('node:child_process')
const { createHash } = require('node:crypto')
const { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { resolve, join } = require('node:path')
const { pathToFileURL } = require('node:url')
const args = process.argv.slice(1)
const audio = resolve(args[args.indexOf('--audio') + 1])
const output = resolve(args[args.indexOf('--output') + 1])
const probeOnly = args.includes('--probe-only')
const profile = mkdtempSync(join(tmpdir(), 'qwen-aec-profile-'))
app.setPath('userData', profile)
app.setPath('sessionData', join(profile, 'session'))
let window
let player
let active
const runs = []
const report = { sourceAudio: audio, sourceSha256: createHash('sha256').update(readFileSync(audio)).digest('hex'),
  runtime: { electron: process.versions.electron, chrome: process.versions.chrome, os: process.getSystemVersion() },
  policy: 'Current output device and system volume are retained. No PWM/Gateway/ASR connection.', runs }
function save() { writeFileSync(join(output, 'comparison.json'), JSON.stringify(report, null, 2) + '\n', { mode: 0o600 }) }
function wav(chunks, rate) {
  const pcm = Buffer.concat(chunks)
  const h = Buffer.alloc(44)
  h.write('RIFF'); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVEfmt ', 8)
  h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22)
  h.writeUInt32LE(rate, 24); h.writeUInt32LE(rate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34)
  h.write('data', 36); h.writeUInt32LE(pcm.length, 40)
  return { bytes: Buffer.concat([h, pcm]), duration: pcm.length / 2 / rate }
}
function cleanup(code) {
  player?.kill()
  rmSync(profile, { recursive: true, force: true })
  app.exit(code)
}
process.once('SIGTERM', () => cleanup(1))
app.on('window-all-closed', () => cleanup(report.status === 'complete' ? 0 : 1))
app.whenReady().then(async () => {
  mkdirSync(output, { recursive: true, mode: 0o700 })
  ipcMain.handle('aec-test:begin', (_event, mode) => {
    if (!['default', 'all'].includes(mode)) throw new Error('Invalid comparison mode')
    active = { mode, chunks: [] }
  })
  ipcMain.on('aec-test:pcm', (_event, data) => {
    if (!active || !data?.byteLength || data.byteLength > 128 * 1024) return
    active.chunks.push(Buffer.from(data))
  })
  ipcMain.handle('aec-test:play', () => new Promise((resolvePlay, rejectPlay) => {
    const startedAt = new Date().toISOString()
    player = spawn('/usr/bin/afplay', [audio], { stdio: 'ignore' })
    player.once('error', rejectPlay)
    player.once('close', code => { player = null; code === 0 ? resolvePlay({ startedAt }) : rejectPlay(new Error(`afplay exited ${code}`)) })
  }))
  ipcMain.handle('aec-test:finish', (_event, details) => {
    const run = { ...details }
    if (active) {
      const result = wav(active.chunks, details.sampleRate)
      const filename = `${active.mode}.microphone.wav`
      writeFileSync(join(output, filename), result.bytes, { mode: 0o600 })
      run.audioFile = filename
      run.durationSeconds = result.duration
      if (!active.chunks.length) run.status = 'no-pcm'
    }
    active = null
    runs.push(run); save()
  })
  // This dedicated window has a minimal test-only bridge, no production APIs or credentials.
  const preload = join(profile, 'preload.cjs')
  writeFileSync(preload, `const {contextBridge,ipcRenderer}=require('electron'); contextBridge.exposeInMainWorld('aecTest',{
    begin:mode=>ipcRenderer.invoke('aec-test:begin',mode), append:data=>ipcRenderer.send('aec-test:pcm',data),
    play:()=>ipcRenderer.invoke('aec-test:play'), finish:details=>ipcRenderer.invoke('aec-test:finish',details)
  });`)
  window = new BrowserWindow({ width: 480, height: 210, title: '麦克风 AEC 对照测试',
    webPreferences: { preload, contextIsolation: true, nodeIntegration: false, sandbox: true, backgroundThrottling: false } })
  window.webContents.session.setPermissionCheckHandler((_wc, permission) => permission === 'media')
  window.webContents.session.setPermissionRequestHandler((_wc, permission, callback, details) => {
    callback(Boolean(permission === 'media' && (!details.mediaTypes || details.mediaTypes.every(type => type === 'audio'))))
  })
  const html = join(profile, 'test.html')
  writeFileSync(html, '<!doctype html><meta charset="utf-8"><title>AEC 对照测试</title><p id="status">正在准备。请保持外放音量和麦克风位置不变，测试期间保持安静。</p>')
  await window.loadFile(html)
  const moduleUrl = pathToFileURL(resolve(__dirname, '../../web/src/realtime/microphone-aec.js')).href
  const results = await window.webContents.executeJavaScript(`(async () => {
    try {
    const { acquireMicrophoneWithDiagnostics, waitForMicrophoneFrames } = await import(${JSON.stringify(moduleUrl)})
    const wait = ms => new Promise(resolve => setTimeout(resolve, ms))
    let deviceId = ''
    const outputs = []
    for (const mode of ['default', 'all']) {
      document.querySelector('#status').textContent = '正在测试 ' + mode + '，请保持安静，音量和设备位置保持不变。'
      const context = new AudioContext({ sampleRate: 48000 })
      await context.resume()
      const { media, diagnostics } = await acquireMicrophoneWithDiagnostics({ mediaDevices: navigator.mediaDevices, mode, deviceId,
        verifyFrames: media => waitForMicrophoneFrames({ media, context }) })
      const track = media.getAudioTracks()[0]
      deviceId ||= track.getSettings().deviceId
      if (mode === 'all' && (diagnostics.appliedMode !== 'all' || diagnostics.allPcmFlow !== 'received')) {
        media.getTracks().forEach(track => track.stop())
        await context.close()
        await aecTest.finish({ mode, diagnostics, status: 'all-unavailable' })
        outputs.push({ mode, status: 'all-unavailable' }); break
      }
      if (${JSON.stringify(probeOnly)}) {
        const received = await waitForMicrophoneFrames({ media, context })
        media.getTracks().forEach(track => track.stop()); await context.close()
        await aecTest.finish({ mode, diagnostics, status: received ? 'frames-received' : 'no-pcm' })
        outputs.push({ mode, status: received ? 'frames-received' : 'no-pcm' }); continue
      }
      const source = context.createMediaStreamSource(media)
      const processor = context.createScriptProcessor(2048, 1, 1)
      const silent = context.createGain(); silent.gain.value = 0
      let recording = false
      processor.onaudioprocess = event => {
        if (!recording) return
        const input = event.inputBuffer.getChannelData(0)
        const pcm = new Int16Array(input.length)
        for (let i = 0; i < input.length; i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767
        aecTest.append(pcm.buffer)
      }
      source.connect(processor); processor.connect(silent); silent.connect(context.destination)
      await wait(1000)
      await aecTest.begin(mode)
      recording = true
      const playback = await aecTest.play()
      await wait(1500)
      recording = false
      processor.disconnect(); source.disconnect(); silent.disconnect()
      media.getTracks().forEach(track => track.stop())
      await context.close()
      await aecTest.finish({ mode, diagnostics, playback, sampleRate: context.sampleRate, status: 'recorded' })
      outputs.push({ mode, status: 'recorded' })
      await wait(1000)
    }
    return outputs
    } catch (error) { return { error: { name: error?.name || 'Error', message: error?.message || String(error), constraint: error?.constraint || '' } } }
  })()`, true)
  if (results.error) throw Object.assign(new Error(results.error.message), { name: results.error.name })
  report.status = results.some(run => run.status === 'all-unavailable') ? 'all-unavailable'
    : runs.some(run => run.status === 'no-pcm') ? 'inconclusive' : probeOnly ? 'probe-complete' : 'complete'
  report.microphonePermission = systemPreferences.getMediaAccessStatus('microphone')
  save()
  console.log('AEC comparison:', report.status, 'Report:', join(output, 'comparison.json'))
  cleanup(0)
}).catch(error => {
  report.status = 'failed'; report.microphonePermission = systemPreferences.getMediaAccessStatus('microphone'); report.error = { name: error?.name || 'Error', message: error?.message || String(error) }
  save(); console.error('AEC comparison failed:', report.error); cleanup(1)
})
