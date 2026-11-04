#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)
const audioIndex = args.indexOf('--audio')
const outputIndex = args.indexOf('--output')
if (audioIndex < 0 || !args[audioIndex + 1]) {
  console.error('Usage: node scripts/compare-microphone-aec.mjs --audio /path/to/playback.wav [--output /path/to/results]')
  process.exit(1)
}
if (process.platform !== 'darwin') throw new Error('This external-player comparison currently requires macOS afplay')
const audio = resolve(args[audioIndex + 1])
if (!existsSync(audio)) throw new Error('Playback audio file not found')
const output = outputIndex < 0 ? resolve(root, 'tmp/microphone-aec-comparison', String(Date.now())) : resolve(args[outputIndex + 1])
mkdirSync(output, { recursive: true, mode: 0o700 })
const child = spawn(resolve(root, 'node_modules/.bin/electron'), [
  resolve(root, 'desktop/experiments/microphone-aec-comparison.cjs'), '--audio', audio, '--output', output, ...(args.includes('--probe-only') ? ['--probe-only'] : []),
], { cwd: root, env: { ...process.env, ELECTRON_DISABLE_SECURITY_WARNINGS: 'true' }, stdio: 'inherit' })
const timer = setTimeout(() => child.kill('SIGTERM'), 180_000)
child.once('exit', code => { clearTimeout(timer); process.exitCode = code ?? 1 })
child.once('error', error => { clearTimeout(timer); console.error(error.message); process.exitCode = 1 })
