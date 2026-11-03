import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'

const run = promisify(execFile)
const here = dirname(fileURLToPath(import.meta.url))

export async function buildSystemAudio({ arch = process.arch, outputDirectory = resolve(here, 'bin', arch) } = {}) {
  if (process.platform !== 'darwin') return null
  if (!['arm64', 'x64'].includes(arch)) throw new Error(`Unsupported system audio helper architecture: ${arch}`)
  const source = resolve(here, 'system-audio/main.swift')
  const plist = resolve(here, 'system-audio/Info.plist')
  const output = resolve(outputDirectory, 'qwen-system-audio')
  const input = await Promise.all([readFile(source), readFile(plist), readFile(fileURLToPath(import.meta.url))])
  const signature = createHash('sha256').update(arch).update(Buffer.concat(input)).digest('hex')
  try {
    if ((await readFile(`${output}.sha256`, 'utf8')) === signature && (await stat(output)).isFile()) return output
  } catch { /* first build */ }
  await mkdir(outputDirectory, { recursive: true })
  await run('xcrun', ['swiftc', '-swift-version', '5', '-O',
    '-module-cache-path', resolve(outputDirectory, 'module-cache'),
    '-target', `${arch === 'x64' ? 'x86_64' : 'arm64'}-apple-macosx14.2`,
    source, '-o', output, '-Xlinker', '-sectcreate', '-Xlinker', '__TEXT',
    '-Xlinker', '__info_plist', '-Xlinker', plist], { timeout: 180000, maxBuffer: 1024 * 1024 })
  await writeFile(`${output}.sha256`, signature)
  return output
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const output = await buildSystemAudio({
    arch: process.argv[2] || process.arch,
    ...(process.argv[3] ? { outputDirectory: resolve(process.argv[3]) } : {}),
  })
  if (output) process.stdout.write(`${output}\n`)
}
