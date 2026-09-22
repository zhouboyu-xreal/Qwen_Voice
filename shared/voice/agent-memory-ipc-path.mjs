import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

// macOS limits sockaddr_un paths to roughly 104 bytes.  Agent Memory state
// directories often include a checkout and user-specific path, so a socket
// placed below that directory is not reliable.  Both Node owners use this
// deterministic, private per-state-directory name under the system temp dir.
export function resolveAgentMemoryTranscriptIpcSocketPath({
  stateDirectory,
  configuredPath = '',
} = {}) {
  const explicit = String(configuredPath || '').trim()
  if (explicit) return resolve(explicit)
  const state = String(stateDirectory || '').trim()
  if (!state) throw new Error('Agent Memory IPC requires a state directory')
  const digest = createHash('sha256').update(resolve(state)).digest('hex')
  return join(tmpdir(), `qwaudio-am-${digest.slice(0, 24)}.sock`)
}
