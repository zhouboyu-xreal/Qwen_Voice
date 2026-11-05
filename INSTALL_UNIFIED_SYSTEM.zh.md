# Qwen Voice Agent：Agent 安装指引

本文供同事的安装 Agent 执行，范围是：**拉取代码、安装依赖、准备本地模型、编译并安装 App、验证安装产物**。

云端实时语音、`llm_client`、`embedding_client` 的配置，以及本人参考音频录制和日常操作，统一见 [首次配置与使用说明](FIRST_RUN_AND_USAGE.zh.md)。安装时保留用户已有的这些设置；新机器不需要先拿到云端 API Key 才能准备依赖和本地模型。

目标仓库：`https://github.com/zhouboyu-xreal/Qwen_Voice.git`；分支：`unified_system`；应用名称：**Qwen Audio Agent**。本文以 macOS 为范围。

## 1. 安装任务与完成标准

Agent 应完成以下工作，而非只把命令交给同事：

1. 检查系统、架构、工具链及已有安装。
2. 获取正确分支并安装 Node/Python 依赖。
3. 为实际使用者准备离线语音模型，验证能加载。
4. 配置外部 Python 与自动路径，不覆盖已有模型凭据和用户数据。
5. 构建 DMG，安装并打开本次产物。
6. 交付实际路径、版本、本地模型准备结果和使用者手册。

安装通过不等于首次配置完成。新用户尚未填写云端 Key、尚未录制参考音频时，应交付“安装完成，待首次配置”，而不是宣称实时对话和 PWM 提炼已验证。

执行约定：

- 使用绝对路径；不同工具调用可能不共享 shell 变量，需要重新声明。
- 复用正确的已有 checkout 和 Python 环境，保留本地改动、有效配置、数据及缓存。不要强制 reset 或清空记忆。
- 修改现有配置前备份，按键合并，不覆盖整个文件，也不输出其中的 Key。
- 在将来更新的 checkout 中，参数和模型选择以当前代码及 `memory/config.yaml` 为准；不为匹配文档修改业务代码。
- 遇到需同事操作的 GitHub 登录、系统安装器或系统权限，说明具体阻塞点，继续其他不受影响的安装步骤。

## 2. 检查环境

```bash
sw_vers
uname -m
git --version
node --version
npm --version
python3.11 --version
xcode-select -p
xcrun swiftc --version
```

| 项目 | 要求 |
| --- | --- |
| 系统 | 双轨系统音频采集需要 macOS 14.2+；较低版本交付时注明仅支持麦克风录音模式 |
| Node/npm | 满足根目录 `package.json` 的 `engines`，当前为 Node `^22.22.2 || ^24.15.0 || >=26.0.0`、npm `>=10` |
| Python | 独立的 Python 3.11 环境 |
| 编译工具 | Command Line Tools/SDK 能运行 `xcrun swiftc` 并编译项目 helper |
| 网络 | 能访问仓库、npm/Python 包源、项目所用模型源 |

缺少编译工具时执行 `xcode-select --install`，等待系统安装完成后再编译。网络或包解析失败需保留原始错误，不能将该步骤视为成功。

## 3. 拉取或更新代码

首次克隆，按本机情况选择工作目录：

```bash
QWEN_VOICE_WORKSPACE="$HOME/Projects"
mkdir -p "$QWEN_VOICE_WORKSPACE"
cd "$QWEN_VOICE_WORKSPACE"
git clone --branch unified_system --single-branch \
  https://github.com/zhouboyu-xreal/Qwen_Voice.git Qwen_Voice
```

设置实际源码目录并核对：

```bash
QWEN_VOICE_REPO="/absolute/path/to/Qwen_Voice"
cd "$QWEN_VOICE_REPO"
git remote -v
git branch --show-current
git status --short
git rev-parse HEAD
```

已有 checkout 时先检查本地修改。确认适合更新后执行：

```bash
cd "$QWEN_VOICE_REPO"
git fetch origin unified_system
git switch unified_system
git pull --ff-only origin unified_system
```

记录实际 commit 和参与构建的本地修改。仓库已包含 `memory/`，不另行拉取 `agent_memory`。SSH 认证失败时核对账号和密钥；HTTPS 也需要仓库权限。

## 4. 安装运行依赖

### Node 依赖

```bash
cd "$QWEN_VOICE_REPO"
npm ci
```

保留开发依赖，不使用 `--omit=dev`。`prepare` 生命周期会构建前端；`npm ci` 必须成功后再打包。

### Python 依赖

优先检查已有 Python 3.11 环境。没有合适环境时，建议在仓库外创建：

```bash
cd "$QWEN_VOICE_REPO"
QWEN_PWM_VENV="$HOME/.local/share/qwaudio/pwm-python"
python3.11 -m venv "$QWEN_PWM_VENV"
QWEN_PWM_PYTHON="$QWEN_PWM_VENV/bin/python"
"$QWEN_PWM_PYTHON" -m pip install --upgrade pip setuptools wheel
"$QWEN_PWM_PYTHON" -m pip install -r memory/requirements.txt
"$QWEN_PWM_PYTHON" -m pip check
"$QWEN_PWM_PYTHON" -c "import sys; print(sys.executable)"
"$QWEN_PWM_PYTHON" -c "import torch, torchaudio, transformers, qwen_asr, funasr, modelscope, soundfile, yaml; print('PWM Python imports OK')"
```

复用 Conda/micromamba 时，用其实际解释器替换 `QWEN_PWM_PYTHON`。后续模型准备和 App 必须使用同一个环境；不要将 venv 软链接解析成基础 Python 来配置。

依赖以 `memory/requirements.txt` 为准。安装冲突不能用 `--no-deps` 绕过；定位具体约束后处理，仍无法完成则明确交付阻塞项。

## 5. 准备本地语音模型

### 5.1 需要准备哪些模型

| 用途 | 当前默认模型/来源 | 准备方式 |
| --- | --- | --- |
| 桌面唤醒词 | sherpa-onnx 中文/英文 Zipformer KWS | 项目 `ensureWakeWordModel` 下载、校验和解包 |
| 本人参考音频采集 VAD | `silero_vad.onnx` | 项目 `ensureVoiceReferenceVadModel` 下载并校验 |
| 离线 ASR | `Qwen/Qwen3-ASR-0.6B` | 依据当前语音 YAML 初始化 `VoiceRuntime`，使用 Hugging Face 缓存 |
| 离线语音 VAD | Python `silero-vad` 提供的模型资源 | 随依赖安装，由 `VoiceRuntime` 验证加载 |
| 说话人识别 | `iic/speech_campplus_sv_zh-cn_16k-common` | `VoiceRuntime` 经项目 ModelScope 加载器准备 |

只准备当前启用的 ASR 后端，不把 YAML 中所有可选后端都下载。云端实时模型、PWM 语言模型和 embedding 使用服务 API，不属于本地权重下载任务。

**以运行 App 的同一个用户执行模型准备，不使用 root 的缓存目录。** 若有自定义模型缓存环境，确保下载命令和 App 的有效环境一致；不要只为下载命令临时切换缓存目录。下列命令按默认缓存配置编写。

### 5.2 下载桌面唤醒词和参考音频 VAD

在仓库根目录执行现有模型管理器，复用程序本身的文件名、缓存结构和 SHA256 校验：

```bash
cd "$QWEN_VOICE_REPO"
node --input-type=module <<'JS'
import { homedir } from 'node:os'
import { resolve } from 'node:path'
import { desktopClientPaths } from './desktop/src/client-paths.mjs'
import { ensureWakeWordModel } from './desktop/src/wake-word/model-manager.mjs'
import { ensureVoiceReferenceVadModel } from './desktop/src/voice-reference/vad-model-manager.mjs'
import { createSherpaWakeWordDetector } from './desktop/src/wake-word/sherpa-detector.mjs'
import { createVoiceReferenceVad } from './desktop/src/voice-reference/vad-runtime.mjs'

const paths = desktopClientPaths(resolve(homedir(), 'Library/Application Support/Qwen Audio Agent'))
console.log('Wake-word model:', await ensureWakeWordModel(paths.wakeWordModelDirectory))
console.log('Reference VAD:', await ensureVoiceReferenceVadModel(paths.voiceReferenceVadModelDirectory))
const wake = await createSherpaWakeWordDetector({ modelRoot: paths.wakeWordModelDirectory })
wake.reset()
const vad = await createVoiceReferenceVad({ modelRoot: paths.voiceReferenceVadModelDirectory })
vad.close()
console.log('Desktop voice models loaded successfully')
JS
```

默认缓存位于 App 用户目录下的 `cache/models/wake-word/` 和 `cache/models/voice-reference-vad/`。若部署采用自定义 Electron userData，应将代码中的根目录改为该 App 实际目录。

首次下载会执行校验；已有完整缓存会复用。不要手工改名模型文件或写一个空文件假装模型已准备好。

### 5.3 准备离线 ASR、VAD 和声纹模型

使用实际 Python 解释器，构造项目现有语音运行时，完成模型下载和加载验证：

```bash
cd "$QWEN_VOICE_REPO"
QWEN_PWM_PYTHON="$HOME/.local/share/qwaudio/pwm-python/bin/python"
PYTHONPATH="$PWD/memory/src" "$QWEN_PWM_PYTHON" - <<'PY'
import logging
from config import load_voice_runtime_config
from voice.voice_runtime import VoiceRuntime

logging.basicConfig(level=logging.INFO)
config = load_voice_runtime_config('memory/config.yaml')
runtime = VoiceRuntime(config, logger=logging.getLogger('install.voice_models'))
try:
    print('ASR backend:', runtime.asr_config.get('asr_backend'))
    print('ASR model:', runtime.asr_config.get('model_id'))
    print('Speaker model:', runtime.speaker_config.get('model_id'))
    print('Offline voice models loaded successfully')
finally:
    runtime.close()
PY
```

若部署使用外部自定义 YAML，应将脚本中的配置路径替换为同一个文件。该步骤初始化语音模型，不打开麦克风、不创建 `PersonWorldModel`，也不向记忆数据库写入数据；它不验证云端 PWM 客户端。

ModelScope 加载器优先检查完整本地缓存，下载失败会尝试国际站。ASR、tokenizer、声纹模型全部成功加载才算通过；仅看到文件夹或部分权重不足以判定成功。记录实际模型和缓存位置，不承诺固定下载耗时。

可以在相同环境下加入 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 重新执行上面的 Python 命令，检查 Hugging Face 部分是否能使用本地缓存；这不表示云端功能可以离线使用。

## 6. 配置安装所需的路径

默认共享配置：`~/.config/qwaudio/config.env`。先检查是否通过 `QWAUDIO_CONFIG_DIR` 指定了其他目录。对实际文件备份后按键合并：

```dotenv
QWEN_AUDIO_MEMORY_PROVIDER=agent-memory
AGENT_MEMORY_PYTHON="/absolute/path/to/the-verified-python"
AGENT_MEMORY_SIDECAR=auto
AGENT_MEMORY_CONFIG=auto
AGENT_MEMORY_STATE_DIR=auto
```

不要把示例占位符写入配置；`AGENT_MEMORY_PYTHON` 使用已验证的绝对路径，不支持 `auto` 或字面量 `~`。保持配置权限为 0600。

- 安装包内置 PWM 代码、默认 YAML，`auto` 会在安装版使用 `Contents/Resources/runtime/memory/`。
- 外部 Python 和依赖必须保留；App 不内置它们。
- 默认 PWM 数据目录为 `~/.config/qwaudio/data/memory/agent-memory`；设置 `QWAUDIO_DATA_DIR` 时使用其下的 `memory/agent-memory`。
- 有效的自定义状态目录、YAML、其他 sidecar 覆盖项和现有 Key 应保留。若要恢复内置代码，先确认显式覆盖是否有意设置。
- `<repo>/tmp/qwen_audio_agent_online` 是可选测试目录，不是默认数据位置。不要在更新时例行清空或迁移它。

不在本步骤替用户配置云端模型或录制本人声音，完成安装后按使用者手册引导。

## 7. 编译、安装与启动

```bash
cd "$QWEN_VOICE_REPO"
npm run desktop:build:local
```

确认命令成功，`dist/desktop/` 生成当前架构的 DMG。该命令使用本地 ad-hoc 签名并关闭公证，不需要正式发布证书。

`desktop/native/before-pack.cjs` 会按架构编译或复用 `desktop/native/bin/<arch>/qwen-system-audio`，并打包到 App 的 `Contents/Resources/native/qwen-system-audio`；无需在用户安装 App 时重新编译。

Agent 随后完成：

1. 如有旧 App，先正常停止录音、等待相关任务结束并退出，保留可恢复的旧安装副本。
2. 挂载本次 DMG，将 App 安装到 `/Applications/Qwen Audio Agent.app`；使用实际挂载路径，可通过 UI 工具或允许的 `hdiutil`、`ditto` 完成。
3. 启动“应用程序”中的安装版，确认不是旧进程、源码 Electron 或 DMG 内副本。
4. 确认能打开设置，并检查 App 内 PWM 文件和当前架构 helper。新用户没有填写实时模型 Key 时进入首次配置界面是预期结果。

若系统要求允许打开或权限授权，请同事完成必要的系统操作。麦克风实录、模型服务连接和声音样本采集交给首次配置流程，不作为缺少云端 Key 时的安装阻塞。

源码调试可使用 `npm run build` 后执行 `npm run desktop`，但不能替代安装版启动检查。

## 8. 安装验收、更新与交付

### 安装验收

| 项目 | 必须记录的结果 |
| --- | --- |
| 源码 | 仓库、分支、commit，以及参与构建的本地修改 |
| 依赖 | Node/npm 版本；实际 Python 路径、版本、pip check 和导入结果 |
| 本地模型 | 唤醒词、参考 VAD、ASR、语音 VAD、声纹模型各自的准备结果与缓存位置 |
| 安装产物 | DMG 路径、App 路径、版本和架构；helper 与内置 PWM 文件存在 |
| 启动 | 本次安装版能打开并进入设置；如果已有完整配置，可额外检查 Gateway 启动 |
| 保留项 | 实际 config.env、PWM 状态目录、旧安装/配置备份位置 |

未成功下载或加载的模型必须列为未完成项，附具体错误及可执行的下一步。不要把“首次运行时再下载”写成已经完成模型准备。

### 更新已有安装

保留实际 Python、配置、数据库及模型缓存。旧 App 和共享同一状态目录的 Gateway 正常退出后，如需备份，复制完整状态目录（包含 SQLite WAL），再更新代码、依赖和模型检查，重新构建并替换 App。不要以清空记忆、删除所有缓存或重命名整个 `tmp/` 作为例行更新步骤。

### 交付给同事

提供一份简短记录：安装完成或具体阻塞、源码 commit、App/DMG 路径、Python 路径、本地模型准备结果、实际配置与数据目录、保留的自定义设置，以及 [首次配置与使用说明](FIRST_RUN_AND_USAGE.zh.md)。

尚未配置云端服务时明确交付：**“安装完成，本地模型已准备；下一步请按使用者手册完成语音前台、PWM 客户端配置及参考音频录制。”** 不输出真实 Key，也不声称已完成实录或记忆提炼验收。
