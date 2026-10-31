# Qwen Voice Agent Unified System 安装指引

本文面向协助安装的 Agent。目标是从包含 Agent Memory、主动唤醒、Pre-wake Context 与环境录制功能的 `unified_system` 分支构建并运行 Qwen Voice Agent Desktop。

不要再单独拉取 `agent_memory` 仓库，也不要配置指向旧仓库的 `AGENT_MEMORY_SIDECAR`、`AGENT_MEMORY_CONFIG` 或 `AGENT_MEMORY_STATE_DIR`。当前分支中的 `memory/` 是唯一应使用的记忆实现。

本文以 macOS 为例；命令中的 `<workspace>`、`<repo>` 均应替换为实际的绝对路径。

## 1. 前置条件

- macOS（本地 Desktop 构建流程）；
- Git；
- Node.js `22.22.2+`（推荐当前 Node 22 LTS）与 npm `10+`；
- Python `3.11`；
- 可访问 DashScope、DeepSeek 与 GLM embedding 服务的网络及对应 API Key。

首次使用会下载 ASR、VAD、说话人识别等模型，需预留足够的磁盘、内存和下载时间。

可先验证运行时：

```bash
node --version
npm --version
python3.11 --version
```

## 2. 拉取唯一源码仓库

在计划存放项目的目录执行：

```bash
mkdir -p <workspace>
cd <workspace>
git clone --branch unified_system --single-branch \
  git@github.com:zhouboyu-xreal/Qwen_Voice.git qwen-voice-agent
cd qwen-voice-agent
git branch --show-current
```

最后一条命令必须输出 `unified_system`。后续所有相对路径均以这个 `qwen-voice-agent` 目录为根目录。

## 3. 备份旧 `tmp` 数据（必做）

本分支的 Agent Memory SQLite schema 已变更，**不能将旧版本的 `tmp/` 直接复用**。如果仓库根目录中已存在 `tmp/`，请先整体改名备份；不要删除它：

```bash
cd <repo>
if [ -d tmp ]; then
  mv tmp "tmp.backup-$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p tmp/qwen_audio_agent_online
```

如需保留旧记忆以供排查，应从备份目录中只读查看旧 `memory.db`，不要将它复制回新的 `tmp/qwen_audio_agent_online`。新版本会在首次运行时创建新的数据库与日志。

## 4. 安装 Node 与 Python 依赖

```bash
cd <repo>
npm install

python3.11 -m venv memory/.venv
memory/.venv/bin/python -m pip install --upgrade pip setuptools wheel
memory/.venv/bin/pip install -r memory/requirements.txt
```

`memory/.venv` 必须保留在源码目录中：安装后的 Desktop 通过下面的配置调用该 Python 解释器和 sidecar。不要只执行 `npm install`，否则 Desktop 可以启动但 Agent Memory、Pre-wake Context 和环境录制无法运行。

## 5. 配置共享运行环境

Desktop 与 CLI/TUI 默认共享配置文件：

```text
~/.config/qwaudio/config.env
```

若文件不存在，可先执行一次 `qwenaudio config`，或手动创建其父目录和文件。将下列内容写入该文件，并将 `<repo>` 替换为第 2 步得到的**绝对路径**：

```dotenv
# Qwen Realtime 前台模型
DASHSCOPE_API_KEY=<your-dashscope-key>
QWEN_AUDIO_REALTIME_MODEL=qwen-audio-3.0-realtime-plus

# 使用仓库内置的 Agent Memory，而不是 markdown / voicemem。
QWEN_AUDIO_MEMORY_PROVIDER=agent-memory

# Python memory runtime：Desktop 安装包不内嵌此 Python 环境，以下绝对路径必填。
AGENT_MEMORY_PYTHON=<repo>/memory/.venv/bin/python
AGENT_MEMORY_SIDECAR=<repo>/memory/integrations/qwen_audio_agent/agent_memory_sidecar.py
AGENT_MEMORY_CONFIG=<repo>/memory/config.yaml
AGENT_MEMORY_STATE_DIR=<repo>/tmp/qwen_audio_agent_online

# Agent Memory 的事实提炼与 embedding 服务。
DEEPSEEK_API_KEY=<your-deepseek-key>
GLM_API_KEY=<your-glm-key>

# 可选：开启唤醒前 60 秒本地 ASR 上下文。
AGENT_MEMORY_PREWAKE_CONTEXT_ENABLED=true

# 可选：开启 Desktop 环境音频录制按钮及其每 10 分钟一次的离线 ASR。
AGENT_MEMORY_AMBIENT_RECORDING_ENABLED=true
# AGENT_MEMORY_AMBIENT_RECORDING_CHUNK_SECONDS=600
```

注意：

- 不要保留任何指向旧 `agent_memory` 工程的 `AGENT_MEMORY_*` 路径；配置中同名环境变量会覆盖仓库内默认路径。
- `AGENT_MEMORY_PREWAKE_CONTEXT_ENABLED` 与 `AGENT_MEMORY_AMBIENT_RECORDING_ENABLED` 可按需设为 `false`；主记忆功能不依赖它们。
- 不必单独配置 pre-wake 或环境录制 sidecar。它们会从 `AGENT_MEMORY_SIDECAR` 的同级目录自动解析。
- `config.env` 含 API Key，不应提交到 Git、粘贴到工单或共享给他人。

## 6. 构建并安装 macOS Desktop

```bash
cd <repo>
npm run desktop:build:local
```

构建完成后，在 `dist/desktop/` 找到生成的 `.dmg`，打开后将 `Qwen Audio Agent.app` 拖入“应用程序”目录。首次打开如被 macOS 拦截，可在 Finder 中按住 Control 点击应用并选择“打开”。

本地构建包仍依赖第 4 步创建的 `<repo>/memory/.venv` 和第 5 步配置的外部源码路径。因此安装 App 后不要移动、删除或重命名 `<repo>`；如必须移动，请同步更新 `config.env` 中所有 `AGENT_MEMORY_*` 路径。

## 7. 首次验证

1. 启动 Desktop，授予麦克风权限；确认能完成一次普通实时语音对话。
2. 在 Desktop 设置中确认使用 DashScope Realtime 前台模型。
3. 说出唤醒词并发起一轮对话；若启用了 Pre-wake Context，可检查其 sidecar 日志。
4. 点击悬浮球的环境录制按钮，开始和停止一次短录制；原始 WAV 与离线 ASR 结果会通过 Agent Memory sidecar 进入同一记忆流水线。

常用诊断位置：

```text
<repo>/tmp/qwen_audio_agent_online/agent-memory-sidecar.log
<repo>/tmp/qwen_audio_agent_online/prewake-context-sidecar.log
<repo>/tmp/qwen_audio_agent_online/ambient-audio-recording-sidecar.log
<repo>/tmp/qwen_audio_agent_online/ambient-recordings/
```

Agent Memory 会按本地 owner 分区保存数据库；日志中出现 `MemoryRuntime` 初始化、`memory_store` 提交或 sidecar `ready` 表明 Python 记忆链路已经启动。

## 8. 常见故障

### Desktop 能打开，但没有记忆、Pre-wake 或环境录制日志

优先检查 `~/.config/qwaudio/config.env` 中：

- `QWEN_AUDIO_MEMORY_PROVIDER=agent-memory`；
- `AGENT_MEMORY_PYTHON` 指向可运行且已安装 `memory/requirements.txt` 的 Python；
- `AGENT_MEMORY_SIDECAR`、`AGENT_MEMORY_CONFIG` 与 `AGENT_MEMORY_STATE_DIR` 都是存在的绝对路径；
- 修改配置后已完全退出并重启 Desktop。

可在终端验证 Python 环境：

```bash
<repo>/memory/.venv/bin/python --version
<repo>/memory/.venv/bin/python -c "import torch, transformers, qwen_asr"
```

### 首次 ASR 很慢

首次启动会下载或加载 ASR/VAD/说话人模型，环境音频首次封存时也会初始化离线 ASR。等待模型缓存完成后再判断性能问题。

### 更新到本分支后报 SQLite 字段或表错误

说明旧数据库仍在被使用。关闭 Desktop，确认 `AGENT_MEMORY_STATE_DIR`，将其中对应的旧 `tmp/` 目录改名备份后重新启动；不要尝试用旧 schema 数据库覆盖新建数据库。

### 环境录制按钮没有效果

确认 `AGENT_MEMORY_AMBIENT_RECORDING_ENABLED=true`，并查看 `ambient-audio-recording-sidecar.log`。该功能依赖 Agent Memory 主 sidecar 的本地 IPC，因此主 sidecar 未启动时环境录制也无法完成 ASR 入库。

## 9. 更新流程

```bash
cd <repo>
git fetch origin unified_system
git switch unified_system
git pull --ff-only origin unified_system
npm install
memory/.venv/bin/pip install -r memory/requirements.txt
npm run desktop:build:local
```

如果更新说明包含 Agent Memory schema 变更，先重复第 3 步备份旧 `tmp/`，再运行新版 Desktop。
