import { createHash } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { dirname, resolve } from 'node:path'

export const VOICE_REFERENCE_VAD_MODEL_FILE = 'silero_vad.onnx'
export const VOICE_REFERENCE_VAD_MODEL_URL = (
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx'
)
export const VOICE_REFERENCE_VAD_MODEL_SHA256 = (
  '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6'
)

const preparations = new Map()

async function downloadModel(path, fetchImpl) {
  const response = await fetchImpl(VOICE_REFERENCE_VAD_MODEL_URL)
  if (!response.ok) {
    throw new Error(`本人声音采集 VAD 模型下载失败：HTTP ${response.status}`)
  }
  const data = Buffer.from(await response.arrayBuffer())
  const digest = createHash('sha256').update(data).digest('hex')
  if (digest !== VOICE_REFERENCE_VAD_MODEL_SHA256) {
    throw new Error('本人声音采集 VAD 模型校验失败')
  }
  writeFileSync(path, data, { mode: 0o600, flag: 'wx' })
}

async function prepare(directory, { fetchImpl }) {
  const target = resolve(directory, VOICE_REFERENCE_VAD_MODEL_FILE)
  if (existsSync(target)) return target
  mkdirSync(dirname(target), { recursive: true, mode: 0o700 })
  const temporary = `${target}.${process.pid}-${Date.now()}.partial`
  try {
    await downloadModel(temporary, fetchImpl)
    renameSync(temporary, target)
    return target
  } finally {
    rmSync(temporary, { force: true })
  }
}

export function ensureVoiceReferenceVadModel(directory, {
  fetchImpl = fetch,
} = {}) {
  const root = resolve(directory)
  const target = resolve(root, VOICE_REFERENCE_VAD_MODEL_FILE)
  if (existsSync(target)) return Promise.resolve(target)
  if (!preparations.has(target)) {
    const pending = prepare(root, { fetchImpl })
      .finally(() => preparations.delete(target))
    preparations.set(target, pending)
  }
  return preparations.get(target)
}
