#!/usr/bin/env bash
# Mirror the Agent Memory runtime into qwen-audio-agent's vendored memory tree.
#
# The two repositories intentionally remain independent Git worktrees.  Run
# this script after changing Agent Memory code that the Qwen Audio Agent
# sidecars use, then commit the resulting changes from qwen-audio-agent.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_TARGET="${SOURCE_ROOT}/../voice_agent/qwen-audio-agent/memory"
TARGET_ROOT="${DEFAULT_TARGET}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/sync_qwen_audio_agent_memory.sh [--dry-run] [--target PATH]

Synchronize the Agent Memory runtime into qwen-audio-agent/memory.

Options:
  --dry-run       Show the planned changes without modifying files.
  --target PATH   Override the qwen-audio-agent memory directory.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --target)
            shift
            if [[ $# -eq 0 || -z "$1" ]]; then
                echo "--target requires a directory path." >&2
                exit 2
            fi
            TARGET_ROOT="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found." >&2
    exit 1
fi

if [[ ! -d "${TARGET_ROOT}" ]]; then
    echo "Target memory directory does not exist: ${TARGET_ROOT}" >&2
    echo "Use --target to specify the qwen-audio-agent memory directory." >&2
    exit 1
fi

SOURCE_ROOT=$(cd "${SOURCE_ROOT}" && pwd)
TARGET_ROOT=$(cd "${TARGET_ROOT}" && pwd)
if [[ "${SOURCE_ROOT}" == "${TARGET_ROOT}" ]]; then
    echo "Source and target must be different directories." >&2
    exit 1
fi

RSYNC_ARGS=(
    --archive
    --delete
    --exclude=__pycache__/
    --exclude='*.pyc'
    --exclude=.DS_Store
)
if [[ ${DRY_RUN} -eq 1 ]]; then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

echo "Synchronizing Agent Memory runtime"
echo "  source: ${SOURCE_ROOT}"
echo "  target: ${TARGET_ROOT}"

# These directories contain all Python code imported by the sidecars, plus the
# integration and test scripts maintained in this repository.  --delete makes
# removed source files disappear from the vendored copy as well; runtime state
# is outside these paths and is never touched.
for directory in src integrations scripts; do
    mkdir -p "${TARGET_ROOT}/${directory}"
    rsync "${RSYNC_ARGS[@]}" "${SOURCE_ROOT}/${directory}/" "${TARGET_ROOT}/${directory}/"
done

# Keep dependency and runtime configuration changes aligned too.  The checked-
# in config contains no credentials; production keys remain environment vars.
rsync "${RSYNC_ARGS[@]}" \
    "${SOURCE_ROOT}/config.yaml" \
    "${SOURCE_ROOT}/requirements.txt" \
    "${TARGET_ROOT}/"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Dry run complete; no files were changed."
else
    echo "Synchronization complete. Review and commit changes in qwen-audio-agent."
fi
