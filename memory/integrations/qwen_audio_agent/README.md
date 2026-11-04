# qwen-audio-agent 集成

`agent_memory_sidecar.py` 将 qwen-audio-agent 的 `MemoryProvider` 协议连接到
本工程的 `MemoryRuntime`。它只接收 Qwen 已确认的转写文本；Realtime 音频、VAD、ASR
和打断仍由 Qwen 处理。

## 配置

在 qwen-audio-agent 的环境文件中设置：

```dotenv
QWEN_AUDIO_MEMORY_PROVIDER=agent-memory
AGENT_MEMORY_PYTHON=/absolute/path/to/python
AGENT_MEMORY_SIDECAR=auto
AGENT_MEMORY_CONFIG=auto
AGENT_MEMORY_STATE_DIR=auto
```

路径留空或设为 `auto` 时，开发版使用当前工程 `memory/`，安装版使用应用资源
`Contents/Resources/runtime/memory/`；Python 代码与默认 YAML 随安装包发布，
不依赖打包机器或源码目录。状态目录默认是 `QWAUDIO_DATA_DIR/memory/agent-memory`
（未覆盖时为 `~/.config/qwaudio/data/memory/agent-memory`），录音也保存在此目录下。

启动时会将当前工程的默认代码路径，以及指向已删除工程的旧标准路径，自动改为
`auto`；修改配置前保留 `config.env.before-agent-memory-auto` 备份。
有效的外部代码路径、自定义 YAML 和已有状态目录保留，缺失的自定义文件仍会报错。
旧工程连同 `tmp/qwen_audio_agent_online` 已被删除时，改用持久数据目录，无法恢复
已删除的记忆或录音。已有状态目录不会自动搬迁；如要迁移数据库，应先退出应用。
Shell 中显式传入的路径只对当前进程解析，不会写回覆盖 `config.env`。
`AGENT_MEMORY_PREWAKE_CONTEXT_SIDECAR` 和 `AGENT_MEMORY_AMBIENT_RECORDING_SIDECAR`
默认跟随主 sidecar，显式设为 `auto` 也遵循同一规则。

`AGENT_MEMORY_PYTHON` 必须是安装了本工程 `requirements.txt` 依赖的解释器。
解释器路径保留用户配置，不随工程路径自动替换。
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

## 桌面双路环境录音

设置 → 应用程序 → 环境录音来源，可选择“麦克风 + 系统音频”（默认）或“仅麦克风”。
此桌面偏好保存在客户端 `settings.env`，对应 `QWEN_AUDIO_AMBIENT_RECORDING_SOURCE`，
值分别为 `microphone-system` / `microphone`；下次开始环境录音时生效。

- 麦克风保持原有采集链路，继续用于唤醒、语音交互和环境录音。
- macOS 14.2+ 使用随应用打包的 Swift / Core Audio Process Tap 辅助程序，直接采集系统播放音频，
  不调用 `getDisplayMedia`、不打开屏幕共享选择器、不创建视频轨。主进程将系统 PCM 直接送入录音
  Sidecar，网页仅控制生命周期和接收进度，不接收系统 PCM；唤醒、实时语音、前唤醒上下文与本人
  声音采集继续仅使用麦克风。Windows 继续使用 Electron loopback；Linux 暂需选择“仅麦克风”。
- 原生辅助程序及主应用均声明 `NSAudioCaptureUsageDescription`。首次使用需授予系统音频录制权限；
  拒绝后可在系统设置的“隐私与安全性”中开启相应的系统音频录制权限。输出设备切换时会停止本次
  录音并提示重新开始，避免将变化后的格式写入旧 WAV。
- `desktop/native/build-system-audio.mjs` 构建辅助程序。开发启动前自动构建当前架构；打包 hook
  分别构建 arm64 / x64，Universal 打包时随应用合并。二进制放在 `.asar` 外的
  `Contents/Resources/native/qwen-system-audio`，由打包签名流程签名。用户运行安装版无需 Swift 工具链。
- 停止时先等待原生程序输出尾段并确认落盘，再请求网页刷新麦克风尾段，最后封存两路文件；
  主进程主动停止、辅助程序异常和退出应用均使用相同的尾段处理。
- 授权取消、缺少有效系统音轨或采集中断均明确报错，不自动降级为麦克风录音。任一来源连续
  10 秒没有交付 PCM 时停止本次环境录音并提示；静音 PCM 不算断流。
- 两路分别保存单声道 PCM16 WAV，文件名包含 `microphone` / `system`，各自保留采样率与时间。
  设备采样率变化或明显时间间断时封存当前文件，避免 WAV 格式错误或压缩时间空隙。
- 两路共用 recordingId，转写证据的 `metadata.audio_source` 标记来源。ASR 使用同一串行 worker，
  转写按采集时间整合后送入唯一的 PWM writer；提交也在 worker 中执行，避免阻塞 PCM 落盘。
  只有两路均处理完成的时间范围才推进覆盖水位；静音批次也通过 `processedThrough` 推进。
  录音停止后提交剩余转写并统一 finalize；失败批次保留 WAV，界面提示转写未完成。
- 本阶段没有增加额外回声消除、混音或跨音轨文本去重。外放可能在两个来源中产生重复内容，
  两路相同文本均保留；不依据录音来源直接认定说话人身份。麦克风原有浏览器音频处理保持沿用。

### 环境录音的双轨共同静音切分

环境录音启动成功后持续保存收到的 PCM，包含静音和背景声，不等检测到人声才开始写入。
录制期间写入 `.partial.wav`，封存后成为 `.wav` 并提交离线 ASR；转写结果再交给 PWM。

- 默认每批达到 **600 秒后** 才寻找切分点，没有强制时长上限。已有
  `AGENT_MEMORY_AMBIENT_RECORDING_CHUNK_SECONDS` / `--chunk-duration-seconds` 参数继续使用，
  语义改为“开始寻找共同静音的最短批次时长”。每次共同切分后重新计时。
- 独立分析线程使用每路各自的 Silero VAD 状态，以采集时间戳寻找同一个连续静音区间，
  不能把不同时间出现的两次静音当成共同静音。默认需连续 **0.8 秒**，且 VAD 未检测到人声、
  原始 PCM 帧的 RMS 不高于 **-35 dBFS**，避免将较响的非人声当成静音。
  开始寻找前预热约一秒，不加载 ASR 或声纹模型，分析复制帧的重采样不改变保存的 PCM。
- 两路使用相同切分时间点封存。快到达音轨已经写过该点的尾部，在磁盘上按采样边界拆入
  下一文件，保留全部采样，不混音、不重复、不裁剪静音，也不为未到达的音轨补造数据。
  两路采样率不同，实际边界最多存在采样取整误差；正常连续录制时批次编号对应。
- 只启用麦克风时，仅等待麦克风静音。两路没有共同静音则继续当前文件；主动停止时直接
  保存尾部，不等待静音。设备格式变化和明显时间间断仍按异常边界封存相关音轨，并重新
  计时和检测，避免跨缺口匹配静音或将不同格式写入同一个 WAV。
- VAD 初始化或推理失败会明确报错，保留已写音频；不会偷偷退回定时硬切。
  `ambient shared batch sealed reason=joint_silence` 日志记录共同切分时间、批次时长和静音阈值。
  录音清单保存 `split_policy`、`shared_split_events`；转写证据记录
  `audio_cut_reason` / `audio_shared_cut_at`。未封存的音频仍等待封存后转写。

可在实际使用的 `AGENT_MEMORY_CONFIG` 中调整内部静音参数：

```yaml
voice_runtime:
  ambient_recording:
    split_silence_seconds: 0.8
    split_quiet_max_rms_dbfs: -35.0
```

这些是录音切分参数，不改变离线 ASR 自身的语音分段设置。重启应用后生效。

### 麦克风 AEC 诊断与外放对照

桌面应用固定优先使用系统级 all，通过 `{ exact: "all" }` 请求麦克风回声消除，且只有
`track.getSettings().echoCancellation === "all"` 才确认生效。设置页不再提供模式选择，
旧的 `QWEN_AUDIO_MICROPHONE_ECHO_CANCELLATION` 配置及页面模式参数不再生效。
普通网页仍沿用 `echoCancellation: true`；独立对照工具继续支持 default/all 两种模式。
不支持 all、实际返回未确认，或严格模式启动后 3 秒内没有音频帧时，释放尝试的音轨并
回退默认模式。静音帧也算有效帧，不会因为环境安静而回退；权限与设备错误仍正常报错。

`desktop.log` 的 `desktop.microphone.capture` 记录实际设备名称、请求/应用模式、
all 能力状态、回退原因、采样率、声道数、NS/AGC 及能力范围。不会记录音频、正文、
原始 deviceId/groupId。每次环境录音还会保存 `<recordingId>.microphone.json`，
与原有录音清单的 `recording_id` 对应；设备切换后的诊断也会追加，最多保留 20 条。
`advertised` 只表示能力列表包含 all，`confirmed` 表示实际设置返回 all；
`rejected`/`unconfirmed`/`no-pcm` 不能当作 all 测试成功。
`allPcmFlow` 记录 all 的音频帧验证结果，`allNegotiatedEchoCancellation` 保留尝试时的实际设置。

macOS 可在工程根目录运行独立对照工具：

```bash
node scripts/compare-microphone-aec.mjs --audio /absolute/path/to/playback.wav
```

工具请求真实麦克风权限，使用外部 `afplay` 重放同一素材，分别采集 default/all。
保持外放音量、输出设备、麦克风位置不变，并在纯回声测试期间保持安静。
两次采集固定同一输入设备、NS/AGC 参数；all 不可确认时跳过第二次播放并明确标记。
工具不连接 Gateway、ASR 或 PWM，不会把测试内容写入实际记忆。
默认输出到 `tmp/microphone-aec-comparison/<timestamp>/`，可用 `--output` 指定目录。
报告包括实际模式、能力、运行时版本、素材 SHA-256 和 WAV；支持 all 只代表模式
协商成功，是否消除了外放仍应听取录音并做同条件比较，不能仅靠该标记判断。

独立工具支持 `--probe-only`：只验证两种模式的音频帧与回退，不播放素材。
没有采集到 PCM 的测试标记为 `no-pcm`/`inconclusive`，不会当作 AEC 消除了所有声音。
macOS 可能把从终端或 Codex 启动的进程归到启动器的麦克风权限；以系统实际授权提示为准。
