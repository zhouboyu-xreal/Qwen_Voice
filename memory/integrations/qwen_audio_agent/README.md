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

## 外部周期任务

桌面应用启动后，由 `desktop/src/periodic-task-controller.mjs` 独立计时。默认每
1800 秒发送一次 `user_state_from_claims` 信号，首次发送在启动一个完整周期后。
配置 `QWEN_AUDIO_PWM_USER_STATE_INTERVAL_SECONDS` 可调整秒数，`0` 可关闭；修改后需重启应用。
图标休眠和环境录音开关不影响计时；退出停止计时，重新启动从零开始，系统恢复后不补发所有错过的周期。

计时器只判断周期，不读取 claim 数据。Gateway 的
`POST /api/memory/periodic-task` 接收 `{ "taskId": "user_state_from_claims" }`，
使用认证身份经 Provider / Sidecar 调用 `PersonWorldModel.trigger_user_state_update()`。
PWM 在接收信号时直接检查已持久化的待处理 claim，无增量立即返回
`queued: false, reason: no_changed_claims`。`queued: true` 表示实际状态更新任务已入队，
不表示模型处理已完成；检查本身不生成独立任务，也不等待正在提炼的新 claim。

有增量则按 `user_state_claim_batch_limit` 提交一批状态更新，由 PWM worker 串行执行；
不再检查周期、最少 claim 数或最长等待时间。
任务成功才确认处理过的 claim 版本；失败和执行期间产生的新版本留待后续周期处理。
PWM 启动、worker 空闲和普通任务完成都不会自动触发周期检查。

离线任务需要处理全部批次时，先等待 claim 提炼任务完成，再显式调用
`drain_pending_user_state_claims()`，然后等待状态更新任务完成。
在线周期信号使用 `trigger_user_state_update()`，离线完整处理使用
`drain_pending_user_state_claims()`，两者均不在 PWM 内部计时。

## 我的活动只读查看

“我的记忆”中的“我的活动”读取当前用户 `memory.db` 的
`memory_user_activity_items`，复用认证后的 `inspect` 路径，不启动提炼任务。

- 列表：`GET /api/pwm/activities?page=1&pageSize=20`。
- 详情：`GET /api/pwm/activities/{id}?evidencePage=1&sourcePage=1`。
- `role` 可选 `actor`、`participant`、`contextual`；空值表示全部角色。
- `from`、`to` 使用带时区的 ISO 时间，查询观察窗口与 `[from, to)` 重叠的记录。
  界面日期按用户本地整天转换，结束日期包含当天。
- 关联事实每页最多 20 条，来源窗口每页最多 6 段；来源不可用时保留缺失提示。

活动表示采集当时观察到的行为、交谈或环境情境。观察时间来自整个来源窗口，
不表示精确活动时长；来源窗口也可能包含其他话题。活动列表不套用意图和状态的
领域、状态或当前/历史筛选。

### 桌面 PWM 模型设置

设置页按“语音前台 → PWM → 后台 Agent → 应用程序”排列。PWM（PersonalWorldModel）
分别配置语言模型 `llm_client` 和向量模型 `embedding_client`，高级参数默认折叠。
服务预设仅填充地址与模型名；Embedding 使用 OpenAI 兼容协议。

本地设置保存在 Gateway 的 `config.env` 中，优先于 `config.yaml` 的
`personal_world_model.llm_model` / `embedding_model`。没有桌面覆盖项时继续沿用 YAML，
包括其中引用的环境变量密钥。对应覆盖项如下：

| 客户端 | 环境变量（前缀均为 `QWEN_AUDIO_PWM_`） |
| --- | --- |
| 语言模型 | `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`、`LLM_TIMEOUT`、`LLM_MAX_TOKENS`、`LLM_THINKING`、`LLM_JSON_MODE` |
| Embedding | `EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_API_KEY`、`EMBEDDING_TIMEOUT`、`EMBEDDING_DIMENSIONS`、`EMBEDDING_BATCH_SIZE`、`EMBEDDING_NORMALIZE` |

密钥分别保存，页面只接收“已配置”状态，留空保留原密钥。连接测试由桌面主进程发送固定测试文本，
不会发送记忆内容、保存草稿或重启运行时。远端 Gateway 模式下，PWM 设置由远端主机管理。
点击“应用”后，本地托管的 Gateway / PWM Sidecar 自动重新加载；其他进程管理的本地 Gateway
需要先停止，再应用 PWM 设置。

Embedding 的模型、请求地址、维度或归一化策略改变后，仅失效可再生成的向量缓存，保留事实、
活动、意图、状态及原文。旧版本没有模型标记的向量也会在升级后重建一次。召回文档向量在共享
工作队列中后台补齐，行动信号向量在后续模式更新时按需补齐；服务暂不可用的缓存会在下次启动时
继续尝试。向量维度必须与服务真实输出一致，不会再补零或截断；重建期间仍可使用词法召回。
