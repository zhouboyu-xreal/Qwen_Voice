# Agent Memory runtime

This directory is the Python memory domain bundled with the Qwen Audio Agent
source tree. It owns durable fact extraction, episodes, entity claims,
prospective objects and recall. The Node Gateway remains responsible for
realtime audio, wake-word state, interruption and function calling.

## Layout

- `src/memory/`: durable memory runtime and SQLite persistence.
- `src/voice/`: local ASR/VAD utilities used by pre-wake context.
- `integrations/qwen_audio_agent/`: JSONL sidecars launched by the Gateway.
- `config.yaml`: source-development configuration. API credentials remain
  environment references such as `${DEEPSEEK_API_KEY}` and `${GLM_API_KEY}`.
- `requirements.txt`: Python dependencies for the sidecars.

## Running from this repository

Install the Python environment once:

```bash
python3 -m venv memory/.venv
memory/.venv/bin/pip install -r memory/requirements.txt
```

Then configure the Gateway with:

```dotenv
QWEN_AUDIO_MEMORY_PROVIDER=agent-memory
AGENT_MEMORY_PYTHON=memory/.venv/bin/python
DEEPSEEK_API_KEY=...
GLM_API_KEY=...
```

`AgentMemoryProvider` automatically uses the in-tree sidecar and
`memory/config.yaml`; `AGENT_MEMORY_SIDECAR` and `AGENT_MEMORY_CONFIG` remain
optional overrides for development. Enable pre-wake context separately with
`AGENT_MEMORY_PREWAKE_CONTEXT_ENABLED=true`.

The desktop release must ship Python code outside `app.asar` and load a
user-owned, credential-free runtime configuration. That packaging policy is
intentionally kept separate from this source-tree integration.
