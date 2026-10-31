# 飞书会议记录提炼测试

入口为 `run_feishu_meeting_extraction.py`。读取飞书导出的 Markdown **文字记录**，通过 `PersonWorldModel` 提炼 memory 和 user state，不执行召回或问答评测，也不生成额外的会议纪要页面。

## 运行方式

在工程根目录、已安装项目依赖的 Python 环境中执行。先只检查解析和身份映射，无需模型凭据：

```bash
python scripts/memory/ambient_transcript/run_feishu_meeting_extraction.py \
  --meeting-file '/Users/byzhou/Downloads/文字记录：EYE 手势 CES DEMO 沟通 2025年12月24日.md' \
  --owner-name '沈劲阳' \
  --parse-only
```

确认 `parsed_meeting.json` 和 `transcript.txt` 后，移除 `--parse-only` 正式运行：

```bash
python scripts/memory/ambient_transcript/run_feishu_meeting_extraction.py \
  --meeting-file '/Users/byzhou/Downloads/文字记录：EYE 手势 CES DEMO 沟通 2025年12月24日.md' \
  --owner-name '沈劲阳'
```

正式提炼使用 `config.yaml` 中的 LLM、embedding 和任务开关；所需环境变量按该配置设置。不额外强制启用原本关闭的 thread 等任务。

连续处理多个文件时，在 `--meeting-file` 后依次列出路径，也可以重复使用该参数：

```bash
python scripts/memory/ambient_transcript/run_feishu_meeting_extraction.py \
  --meeting-file '/path/to/第一次会议.md' '/path/to/第二次会议.md' \
  --meeting-file '/path/to/第三次会议.md' \
  --owner-name '沈劲阳'
```

文件严格按参数传入顺序处理，不按文件名或会议时间重新排序；建议按实际时间先后传入。多个文件使用同一个 PWM 实例、同一组数据库和同一个用户身份，后续会议可以使用前面已经生成的记忆与状态。每份文件独立读取自己的会议时间，独立完成收尾与 state 更新后再处理下一份；不会将不同会议拼成一个输入批次。所有文件都会先完成解析校验，某份解析失败时不会启动提炼；提炼中途失败则停止后续文件，并保留已完成结果和失败报告。

其他参数：

- `--config PATH`：使用指定配置文件。
- `--output-dir PATH`：指定新目录或空目录；非空目录拒绝运行，避免混入旧数据库。
- `--segment-limit N`：每份文件分别只处理按时间排序后的前 N 段发言，默认全部。
- `--session-start '2025-12-24T17:01:00+08:00'`：仅用于单文件，覆盖会议时间零点；文件无法确定日期或开始时间时必须提供。相对发言时间仍会叠加到该零点。未带时区时按 GMT+08 处理。多文件不允许使用同一个时间覆盖所有会议，须确保各文件的日期与开始时间可被解析。
- `--llm-model NAME`、`--llm-base-url URL`：覆盖配置中的模型或服务地址。
- `--log-level DEBUG`：增加诊断输出。

## 身份与时间规则

- `--owner-name` 精确匹配发言标题中的姓名，忽略开头的 `@`。匹配后统一标记为 `speaker_用户本人`。
- 将指定姓名作为测试配置确认的用户别名，在提交输入前写入 `memory_owner_entity_aliases`，因此第三方明确提及该姓名时也能使用现有别名识别流程。该配置不是虚构 fact，不创建支持它的事实记录。
- 其他人的姓名和匿名会议室说话人标签原样保留，不猜测匿名说话人是否也是用户；不替换正文中出现的姓名。
- 保留所有参会者的文本，不只保留用户发言，以便理解提议、回应、分工和依赖。
- 从会议标题/文件名读取日期，从“会议时间”读取开始时间及 GMT 时区。此例的 `00:01:06` 对应 `2025-12-24T17:02:06+08:00`。
- 导出文件只有发言起点，没有真实结束时间。因此每段 `ended_at = started_at`，并在 metadata 标记结束时间未知；不会将下一人开口时间当作上一人的真实结束时间。相同时间的发言保持原文顺序。

## 输出与处理顺序

默认输出目录：`tmp/ambient_transcript/feishu_meeting/<运行时间>/`。

- `parsed_meeting.json`：会议元信息、姓名映射、源文件行号和规范化输入。
- `transcript.txt`：可人工核对的转写文本。
- `run.log`：测试入口日志。
- `memory_manager.log`：PWM 各模块及任务执行日志。
- `memory.db`、`user_model.db`、`task_scheduler.db`：本次独立的记忆、模型与调度数据库。
- `report.json`：提炼结果数量、任务统计及错误诊断；正式运行失败时也尽量保留报告。

单文件保留上述目录布局。多文件时，数据库和日志仍共用根目录；每份会议的 `parsed_meeting.json`、`transcript.txt` 和正式提炼的 `report.json` 放在 `meeting_0001/`、`meeting_0002/` 等子目录中，即使文件同名也不会覆盖。根目录额外保存按输入顺序排列的 `parsed_meetings.json` 索引；总 `report.json` 包含每份会议的状态和全程任务统计。各会议报告中的 `table_counts` 是处理到该会议后的数据库累计数量，不是该文件单独新增的数量。未执行的后续会议在总报告中标为 `pending`。

处理顺序为：登记已知用户身份 → 整批 `accept_memory_input(ambient_recording_enabled=True)` → `finalize_memory_session` → 等待任务完成 → 按最后一段输入时间强制检查 claim → state 窗口 → 再次等待。章节、fact、activity 及后续任务均使用现有调度逻辑。

每份会议通过 `operation_delta` 记录本文件处理期间的任务增量，不将上一场会议的失败计入本场。任务失败或返回 `partial` 时，只在 `failed_tasks` / `partial_tasks` 中记录详情，将该会议标为 `completed_with_failures`，继续处理下一个文件；全部正常时为 `completed`。总报告沿用这两种完成状态，有失败也会正常结束运行，不因任务失败返回非零退出码。输入被拒、finalize 失败、队列无法 drain 或非预期提交拒绝仍会中止，并将总状态标为 `failed`。此行为不改变 Scheduler 的失败数据重试或 pending 策略。

`--parse-only` 不创建数据库，也不执行上述提炼任务。正式运行只是提炼测试，不包含自动的语义准确率评分；需要结合原文、signals、intents、claims 和 states 人工检查结果。

## 离线回归测试

```bash
python -m unittest discover -s tests -p 'test_feishu_meeting_extraction.py' -v
```

覆盖姓名精确匹配、匿名说话人保留、相对时间及时区、跨日、同刻顺序、无效输入、只解析模式、已知别名写库、多文件顺序与独立时间、每文件数量限制、共享 PWM 和数据库、任务失败/partial 后继续下一文件、各会议增量统计，以及输入被拒时停止后续文件等情况，不调用真实模型。
