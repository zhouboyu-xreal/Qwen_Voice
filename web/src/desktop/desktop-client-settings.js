import { resolveOrbSkinId } from '../../../shared/orb-skin-catalog.mjs'
import {
  desktopAutoHideSeconds,
  desktopWakeWordEnabled,
} from './desktop-hide.js'

export function initialDesktopClientSettings(search = '') {
  const params = new URLSearchParams(search)
  return {
    orbSkinId: resolveOrbSkinId({
      orbSkin: params.get('orbSkin'),
      orbStyle: params.get('orbStyle'),
    }),
    autoHideSeconds: desktopAutoHideSeconds(search),
    wakeWordEnabled: desktopWakeWordEnabled(search),
    language: params.get('lang') || '',
    microphoneEchoCancellation: 'all',
  }
}

export function applyDesktopClientSettings(current, update = {}) {
  return {
    orbSkinId: update.orbSkin
      ? resolveOrbSkinId({ orbSkin: update.orbSkin })
      : current.orbSkinId,
    autoHideSeconds: Number.isFinite(update.autoHideSeconds)
      ? update.autoHideSeconds
      : current.autoHideSeconds,
    wakeWordEnabled: true,
    language: update.language || current.language,
    // Desktop capture always requests system-wide AEC, including with legacy settings.
    microphoneEchoCancellation: 'all',
  }
}
