// The memory browser shares the authenticated renderer relay, never the orb's
// presence, microphone, or Gateway voice-client lifecycle.
export function createMemoryWindowController({
  BrowserWindow,
  getUrl,
  getTitle,
  getWorkArea = () => ({ width: 1200, height: 900 }),
  onError = () => {},
}) {
  let window = null
  const current = () => window && !window.isDestroyed() ? window : null
  async function load(target) {
    try {
      await target.loadURL(getUrl())
      return true
    } catch (error) {
      onError(error)
      return false
    }
  }
  return {
    window: current,
    async open() {
      let target = current()
      if (!target) {
        const area = getWorkArea()
        target = new BrowserWindow({
          width: Math.min(980, area.width), height: Math.min(740, area.height),
          minWidth: Math.min(600, area.width), minHeight: Math.min(480, area.height),
          title: getTitle(), show: false, autoHideMenuBar: true,
          webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
        })
        window = target
        target.webContents.setAudioMuted?.(true)
        target.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
        target.webContents.on('will-navigate', (event, value) => {
          const allowed = new URL(getUrl())
          const candidate = new URL(value)
          if (candidate.origin !== allowed.origin || candidate.pathname !== allowed.pathname
            || candidate.searchParams.get('view') !== 'memory') event.preventDefault()
        })
        target.on('closed', () => { if (window === target) window = null })
        await load(target)
      }
      if (target.isDestroyed()) return false
      if (target.isMinimized()) target.restore()
      target.show()
      target.focus()
      return true
    },
    async reload() {
      const target = current()
      if (!target) return false
      target.setTitle(getTitle())
      return load(target)
    },
    close() { current()?.destroy() },
  }
}
