# qwen-audio-agent 集成

`agent_memory_sidecar.py` 将 qwen-audio-agent 的 `MemoryProvider` 协议连接到
本工程的 `MemoryRuntime`。它只接收 Qwen 已确认的转写文本；Realtime 音频、VAD、ASR
和打断仍由 Qwen 处理。

## 配置

在 qwen-audio-agent 的环境文件中设置：

```dotenv
QWEN_AUDIO_MEMORY_PROVIDER=agent-memory
AGENT_MEMORY_PYTHON=/absolute/path/to/python
AGENT_MEMORY_SIDECAR=/absolute/path/to/agent_memory/integrations/qwen_audio_agent/agent_memory_sidecar.py
AGENT_MEMORY_CONFIG=/absolute/path/to/agent_memory/config.yaml
AGENT_MEMORY_STATE_DIR=/absolute/path/to/persistent/agent-memory-state
```

`AGENT_MEMORY_PYTHON` 必须是安装了本工程 `requirements.txt` 依赖的解释器。
Sidecar 会按 Qwen Gateway 的 `ownerId` 进行 SHA-256 分区；每个用户拥有独立的
`memory.db`、去重索引和 `MemoryRuntime` worker。

Sidecar 默认追加运行日志到
`<AGENT_MEMORY_STATE_DIR>/agent-memory-sidecar.log`；可直接传入
`--log-path` 覆盖。日志不写入对话正文，仅包含请求数量、任务提交结果、召回字符数和异常。

## 生命周期

当前版本实现 qwen-audio-agent `MemoryProvider` v2 的 session-observation 路径：

1. Gateway 断开会话时，把已确认的 user/assistant 对话交给 sidecar；
2. Sidecar 调用统一的 `MemoryRuntime.accept_memory_input()`；
3. `flush()` 提交缓冲内容并在同一串行队列中安排 entity-claim update；
4. `query()` 直接调用 `MemoryRuntime.trigger_memory_recall()`，读取最新已提交快照。

因此 recall 不会等待 store/entity-claim 队列完成。Provider 同时维护 Qwen 所需的有界
`user` / `memory` Markdown 快照，以保持显式读取和编辑工具的兼容；自动提炼与语义召回
仍完全由 agent_memory 负责。

逐 turn 写入、原生 PCM 采集、以及面向前台模型的独立 `memory_recall` function tool
尚未在本阶段启用；它们需要对 qwen-audio-agent 的 Provider 生命周期和工具面继续扩展。
