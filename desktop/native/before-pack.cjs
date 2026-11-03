const { Arch } = require('builder-util')

module.exports = async context => {
  if (context.electronPlatformName !== 'darwin') return
  const { buildSystemAudio } = await import('./build-system-audio.mjs')
  await buildSystemAudio({ arch: Arch[context.arch] })
}
