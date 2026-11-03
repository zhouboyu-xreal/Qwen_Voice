/** Main-process policy for the local ambient system-audio track. */
export function systemAudioCaptureSupported(platform, systemVersion) {
  if (platform === 'win32') return true
  if (platform !== 'darwin') return false
  const [major, minor = 0] = String(systemVersion).split('.').map(Number)
  return major > 14 || (major === 14 && minor >= 2)
}

export function configureSystemAudioCapture({ session, desktopCapturer, platform, allowed }) {
  session.setDisplayMediaRequestHandler(async (request, callback) => {
    // macOS uses the native Core Audio helper and must never enter the
    // display-capture API or a screen sharing picker.
    if (platform !== 'win32' || !allowed(request.frame)) return callback({})
    try {
      const sources = await desktopCapturer.getSources({
        types: ['screen'], thumbnailSize: { width: 0, height: 0 },
      })
      if (!allowed(request.frame) || !sources[0]) return callback({})
      callback({ video: sources[0], audio: 'loopback' })
    } catch {
      callback({})
    }
  }, { useSystemPicker: false })
}
