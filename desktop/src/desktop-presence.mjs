export class DesktopPresence {
  constructor({
    getWindow,
    globalShortcut,
    logger,
    onSleep = () => {},
  } = {}) {
    this.getWindow = getWindow
    this.globalShortcut = globalShortcut
    this.logger = logger
    this.onSleep = onSleep
    this.state = 'sleeping'
    this.shortcut = ''
    this.shortcutRegistered = false
    this.shortcutPaused = false
    this.wakeReason = ''
  }

  send(state, reason = '') {
    this.state = state
    const window = this.getWindow?.()
    if (window && !window.isDestroyed()) {
      window.webContents.send('qwen-audio-agent:lifecycle', { state, reason })
    }
  }

  // Showing or focusing the surface never activates microphone forwarding.
  show(reason = 'shortcut') {
    const window = this.getWindow?.()
    if (!window || window.isDestroyed()) return false
    if (window.isMinimized()) window.restore()
    window.show()
    window.focus()
    window.webContents.send('qwen-audio-agent:lifecycle', {
      state: this.state,
      reason: reason === 'wake-word' ? reason : 'activity',
    })
    return true
  }

  wake(reason = 'shortcut') {
    if (reason !== 'wake-word') return this.show(reason)
    const window = this.getWindow?.()
    if (!window || window.isDestroyed() || this.state !== 'sleeping') return false
    this.wakeReason = reason
    this.send('waking', reason)
    return true
  }

  sleep(reason = 'inactivity') {
    if (this.state === 'sleeping') return this.state
    this.wakeReason = ''
    this.send('sleeping', reason)
    this.onSleep()
    this.logger?.info('desktop.sleeping', { reason })
    return this.state
  }

  // Retain the public orb-shell IPC contract; sleep leaves the window visible.
  hide(reason = 'inactivity') {
    return this.sleep(reason)
  }

  ready() {
    if (this.state !== 'waking') return false
    const reason = this.wakeReason || 'unknown'
    this.wakeReason = ''
    this.send('active', 'ready')
    this.logger?.info('desktop.active', { reason })
    return true
  }

  registerShortcut(accelerator) {
    if (this.shortcut === accelerator && this.shortcutRegistered) return true
    const registered = this.globalShortcut.register(
      accelerator,
      () => this.show('shortcut'),
    )
    if (!registered) return false
    if (this.shortcut && this.shortcut !== accelerator) {
      this.globalShortcut.unregister(this.shortcut)
    }
    this.shortcut = accelerator
    this.shortcutRegistered = true
    return true
  }

  pauseShortcut() {
    if (this.shortcut && this.shortcutRegistered) {
      this.globalShortcut.unregister(this.shortcut)
    }
    this.shortcutPaused = true
    this.shortcutRegistered = false
  }

  resumeShortcut() {
    this.shortcutPaused = false
    this.shortcutRegistered = this.registerShortcut(this.shortcut)
    return this.shortcutRegistered
  }

  destroy() {
    this.globalShortcut.unregisterAll()
    this.shortcutRegistered = false
    this.shortcutPaused = false
  }
}
