"""Chinese prompt templates for the unified memory prototype."""

MEMORY_RETRIEVED_FORMAT_PROMPT_ZH = """[统一记忆]
系统说明：召回结果可包含可追溯的 fact 证据、entity claim，以及当前有效的 goal、plan、work item。claim 和 prospective object 是由 facts 支撑的派生记忆，不得将其表述为未经证据支持的新事实。
时间字段说明：对于 fact，dialogue_time 表示对话/转写讨论该 fact 的时间；event_time 表示 fact 描述的现实事件发生时间。二者含义不同；event_time 未知时，不要用 dialogue_time 推断它。
{memory_sections}"""

MEMORY_RETRIEVED_SECTION_SPECS_ZH = (
    (
        "[检索事实]",
        "这些是直接从 memory_facts 检索出的、按相关性排序的叙事事实。",
        "fact",
    ),
    (
        "[实体知识]",
        "这些是当前有效的 entity claim；应结合其支撑事实理解，不要忽略状态或来源。",
        "entity_claim",
    ),
    (
        "[当前目标]",
        "这些是当前有效的 goal，状态和目标时间优先于旧的相关事实。",
        "goal",
    ),
    (
        "[当前计划]",
        "这些是当前有效的 plan，注意计划不等于已经完成。",
        "plan",
    ),
    (
        "[当前事项]",
        "这些是当前未关闭的 work item，注意其责任人、截止时间和状态。",
        "work_item",
    ),
)

ENTITY_EXTRACTION_GUIDANCE_ZH = """实体提取规则:

实体不是只限传统 NER。这里的实体指后续可以跨 facts 聚合、检索、建图的语义锚点。
优先抽取对长期记忆有复用价值的名词或短名词短语，而不是只抽取专有名词。

实体类型(type 可选值):
- PERSON(人): 对话中提到的具体人名、称呼、角色或 speaker
- ORGANIZATION(组织): 公司、团队、机构
- LOCATION(地点): 地理位置、场所
- PRODUCT(产品): 产品名、服务名
- PROJECT(项目): 项目名、产品名或长期工作事项
- TECHNOLOGY(技术): 技术栈、框架、库、工具、API 或系统
- CONCEPT(概念): 抽象概念、方法论、理论或可复用想法
- TOPIC(主题): 讨论的话题领域
- PREFERENCE(偏好): 用户的偏好、喜好、习惯或厌恶
- OTHER(其他): 明确提到但不适合上述类型的实体

应该抽取的实体包括：
- 对话主体或角色：用户、助手、speaker_1、speaker_2、妻子、孩子、团队、客户等
- 用户长期相关的领域、问题、任务、状态或场景：健康管理、身体状态、工作、商务活动、应酬、家庭教育、夫妻沟通、疲劳感等
- 可复用的方案、方法、工具、活动或对象：健康饮食、家庭会议、统一规则、野餐、瑜伽垫等
- 明确影响用户选择的约束对象或条件：经济负担、固定作息、时间不足、工作压力等

不要抽取普通时间表达作为实体，例如：今天、昨天、上周、最近三天、2026-05-07、10:30、三个月。
时间应作为 fact 的时间元数据处理，不进入 entity graph。
只有有语义身份的命名时间概念才可作为实体，例如：春节、Q3 财报季、Sprint 42。

不要抽取纯属性、纯形容词短语、孤立程度词或泛化标签作为实体；它们应保留在 fact text、keywords、topic 或 state 中。
例如：低场地依赖、低强度、高优先级、低成本、强隐私、轻量级。
但如果短语中包含可复用的核心对象或场景，应抽取核心对象，例如：
- “长期高强度工作带来的疲劳感”可抽取“工作”“疲劳感”
- “高频商务活动”可抽取“商务活动”
- “经济负担太重”可抽取“经济负担”

实体应来自对话中明确出现或由角色/事实主体直接确定的内容，不要过度推断。
每条长期记忆 fact 通常至少包含主体实体（如 用户/助手/speaker）和 1-4 个核心语义锚点。"""


EPISODE_SUMMARY_PROMPT_ZH = """你是长期记忆系统的 episode 聚合模块。输入是一段连续时间内的原始交互/转写片段，以及从同一片段中提炼出的高价值 facts。你的任务是生成一段可回顾经历的叙事摘要。

输入角色：
- `source_segments` 是按时间排列的原始证据，是 episode 内容、先后关系和语气的唯一主要依据。
- `evidence_facts` 是从这些片段中筛出的长期价值锚点，只用于核对重要信息是否被遗漏；绝不能逐条改写、拼接或扩写它们。

概念边界：
- fact 是可独立召回的原子证据，保留可精确引用的事实细节。
- episode 是一次连续经历的上层叙事，应回答“这段时间发生了什么、围绕什么展开、最后形成什么结论或未决点”。
- episode 不是长期 state、用户画像、跨经历规律、风险评估或待办事项；不要将一次经历推广成长期结论。

生成规则：
1. 先从原始片段恢复这段经历的主线：背景/触发 → 讨论、观察或行动 → 关键转折/态度 → 结果、决定或未决点。没有证据的环节直接省略。
2. 可以概括多个相关表达，但不要按说话轮次复述，也不要把每条 fact 各写一句。summary 必须比 facts 更连贯、更具上下文，而不是更长的事实清单。
3. 忽略唤醒词、寒暄、礼貌收尾、重复确认、无信息量的短回应和助手的模板化表述。只保留理解这次经历所需的对象、场景、问题、用户关切、关键回应、决定、约束、结果或开放问题。
4. `evidence_facts` 中的高价值信息如与原始片段一致，应尽量在合适的叙事位置覆盖；如两者存在冲突、歧义或 facts 过度概括，以原始片段为准，并保留不确定性。
5. 保留时间顺序、条件和立场。不能把建议写成决定、把可能/计划写成完成、把助手的观点写成用户的信念，也不能编造因果、责任人、截止时间或结果。
6. 若同一连续区间存在多个互不相关但有价值的话题，可用一段有层次的概括串联，但不要为了单一主题删除重要内容或虚构关联。
7. summary 必须是一段简洁、自包含的经历摘要；读者不看 source segments 也能理解核心人物/对象、发生过程和结论/未决点。
8. title 是检索标题：简短、具体、可区分，优先“对象 + 核心事件/问题/决定”，不要使用泛主题标题。
9. canonical_topics 只输出 1-3 个稳定主题。优先复用 evidence_facts 中的 `fact_root_topic`；不要将动作、一次性结论、情绪或零散关键词当作主题。
10. 只返回符合格式的 JSON，不要 markdown、解释文字或额外字段。

输出格式：
{
  "title": "简短具体标题",
  "summary": "一段自包含、按时间和因果组织的 episode 摘要",
  "canonical_topics": ["稳定主题1", "稳定主题2"]
}

原始 source segments：
{source_segments}

高价值 evidence facts（仅用于覆盖核对，不要逐条拼接）：
{evidence_facts}
"""


UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH = """你是 AI 眼镜长期记忆系统的记忆提炼模块。

当前记忆结构：
- fact：从一批连续证据中提炼出的可追溯、自包含、可独立召回的 narrative fact，是记忆提炼的基本证据单元。
- episode：由独立模块根据一段连续时间内持久化的原始交互/转写片段生成的更高层经历摘要；已生成 facts 仅作为重要信息覆盖锚点，它不是每个输入批次或每条 fact 的简单副本。
- memory_topic_items：由已落库 facts 与 episodes 提供的可复用主题命名词表。它包含 canonical topic 和 fact aspect topic，但不保存主题进展、总结或历史事实。
- entity_claim：由独立 reflection 模块从 facts 中形成的、可回溯证据的实体主张；它不是当前对话的直接证据。
- goal / plan / work_item：由独立 Intent & Execution 模块从已落库 facts 提取的未来目标、明确安排和可闭环责任事项；本 prompt 不输出这些对象。

你现在需要从下面按时间顺序排列的对话/转写证据批次中提取 Hindsight 风格的高质量 narrative facts。episode summary、episode canonical_topics 将由独立模块根据持久化的原始片段及已生成 facts 负责，不要在本 prompt 中输出 episode 级字段。

写入资格门槛（先判断，未通过时直接输出空 `facts`；不要为了覆盖输入而生成 fact）：
- 默认输出 0 条 fact。只有同时满足“证据可靠”和“未来可用”时才输出；可被流畅概括不等于值得长期记忆。
- fact 的核心内容必须由用户的语义完整、指代明确的表达，或当前批次中可验证的执行结果支撑。助手的猜测、补全、泛化介绍、安慰、追问、复述和推荐，不能单独证明用户的偏好、身份、情绪、计划、能力或事实。
- “这个/那个/他/她/对/嗯/不去/这就是”等指代不明、语义不完整的短句，只有在当前批次后续的用户表达明确消歧时才能作为证据；不能根据助手的猜测或回答补全其含义。
- 不要把“助手没有理解”“用户没有补充”“问题尚未澄清”本身写成 `open_question` 或其他 fact；除非用户明确要求后续跟进某个对象明确、仍未解决的问题。
- 优先保留：用户明确的稳定身份、偏好、习惯、关系或约束；带具体对象及时间/地点的个人事件或计划；明确决定、承诺、长期指令；用户确认的项目结果、风险或重要问题。
- 可以保留用户明确表达的持续兴趣、困难或目标，但只记录用户事实。助手给出的建议、教程、解释或方案，只有被用户明确接受、选择、执行或成为后续讨论的约束时才可写入。
- 丢弃：唤醒词和寒暄、礼貌确认、浏览或展示过程、重复确认、泛知识讲解、一次模糊提问、未被采纳的建议、纯对话修复、无后续价值的感叹，以及不能独立解释的 ASR 碎片。
- 不要输出被丢弃内容的解释、占位 fact 或低优先级 fact；只返回通过门槛的 facts。

topic 和实体使用规则：
- fact 的 `fact_root_topic` 必须由当前证据中的主要稳定议题产生；`fact_aspect_topic` 应保留该 fact 在根主题下的具体讨论方面。没有充分依据时使用当前证据中的保守、具体 topic。
- `entities` 保留所有与 fact 直接相关的实体，用于完整召回；`primary_entity` 表示这条 fact 主要描述、影响或归属的单一实体，必须来自 `entities`。对于用户自己的偏好、习惯、约束或风险，优先将“用户”作为 primary_entity；对于助手自己的动作或建议，优先将“助手”作为 primary_entity。

memory_topic_items 使用规则：
- `memory_topic_items` 仅是可复用的命名候选，不是当前事实证据，也不承担主题状态更新。不得根据其中的名称补全当前对话未明确支持的对象、进展、结论或关系。
- 只有当前证据的核心对象、讨论目标和语义范围与候选严格对应时，才可原样复用候选名称；同属“旅行”“健康”“产品”等宽泛领域不足以构成对应关系。
- `canonical_topics` 候选优先用于 `fact_root_topic`；`aspect_topics` 候选优先用于 `fact_aspect_topic`。若没有可靠匹配，必须根据当前证据生成保守、具体的新主题，不要强行选择已有名称。
- 不要将动作、一次性结论、情绪、完整句子或零散关键词当作 topic；root topic 表示稳定的主要议题，aspect topic 表示该议题下更具体的讨论方面。

""" + ENTITY_EXTRACTION_GUIDANCE_ZH + """

Hindsight 风格 narrative fact 的核心要求：
- 以下要求只适用于已经通过写入资格门槛的 fact。每条 fact 应覆盖一次完整 exchange 或一个清晰议题片段，而不是单个 utterance。不要把“用户提出问题”“助手给出建议”“用户否定/接受建议”机械拆成多条碎片；如果它们围绕同一问题相互回应，应优先合并成一条 narrative fact。
- 每条 fact 必须能在不阅读原始对话的情况下独立理解，并保留对话的 pragmatic flow：用户为什么提出这个问题，助手给了什么方案，用户如何回应，最后形成了什么倾向、决定、约束、未解决问题或下一步。
- 每条 fact 应在 text 中优先体现 what（完整事件/议题/方案/结论）；when、where、who、why 只有在输入证据明确出现且有助于理解时才加入。缺失的信息直接省略，不要写“未提及具体地点/场景”“没有说明原因”等无信息量的占位句。
- 压缩解释过程，不压缩事实答案；删除无关细节，但不要删除理解事实所需的主体、对象、时间、关键动作、用户态度、结果、决定或约束。
- 对一个 5 轮左右的对话批次或一段多人转写片段，通常输出 1-3 条 facts；只有当批次中确实存在多个互不相关的事件/议题时才拆开。绝大多数情况下不要超过 5 条。

fact_type 判别规则：
- `fact_type` 描述当前 fact 的主要语义作用，只能是 preference、decision、request、recommendation、action、commitment、open_question、risk、error、context、instruction、other。
- 选择最能表达该 fact 对后续记忆价值的一个类别；不要仅因助手未回答或用户表述模糊而使用 `open_question`。

时间保真要求：
- 必须保留影响语义的顺序词和先后关系：first、first time、second、previous、next、later、earlier、before、after、once、again、subsequent、prior、last、most recent，以及“第一次/首次/第二次/之前/之后/此前/随后/后来/更早/最近一次/上一次”等。不要把“first service on March 15”弱化成“service experience”，而应保留“3月15日第一次保养/首次 service”这样的可比较时间锚。
- 必须在 text 和 keywords 中保留相对时间表达：yesterday、last Saturday、previous week、two months ago、about a month ago、mid-February、recently、shortly after，以及“昨天/上周六/前一周/两个月前/约一个月前/二月中旬/最近/不久后”等。如果能根据 Conversation timestamp 无歧义换算，直接将解析后的实际事件时间写入 `event_time_key`。
- 每个片段中的 `Time` 是该片段发生的对话/转写时间，只能作为推导相对事件时间的参考锚点，不是 fact 的默认事件时间。必须结合 fact 所描述的具体事件和原文时间表达，单独推导 `event_time_key`；不要因为 fact 在某个时间被讨论，就把该对话时间直接复制为事件时间。
- `event_time_key` 表示 fact 所描述事件最有代表性的现实发生时间或时间锚点，不表示对话时间、LLM 提炼时间或当前系统时间。它是单一时间字段，不要输出事件结束时间或额外的起止时间字段。
- 时间推导优先级为：原文明确的绝对日期/时间 > 结合片段 `Time` 可以无歧义换算的相对时间 > 明确表示事件就在当前对话中发生的时间。比如对话时间为 2023-05-30，`last month (around April 2023)` 的 `event_time_key` 应为 2023-04 附近的代表性日期，而不是 2023-05-30；`last weekend (May 27-28)` 应使用 2023-05-27 附近的代表性日期；只有“今天决定/刚刚完成”这类明确发生在当前对话中的事件，才使用 2023-05-30。
- 如果 fact 同时描述当前对话行为和更早发生的背景事件，应以该 fact 主要描述的事件为准；必要时拆成多条 facts，不能用对话时间覆盖更早事件。若只能确认月份、周末或相对时间范围，保留原始时间表达在 text/keywords 中，并在 `event_time_key` 中填写保守的代表性时间锚点。
- 优先输出证据中明确给出的日期、时间、星期或相对时间。相对时间只有在结合当前片段的 `Time` 可以无歧义换算时才转换为绝对时间；无法判断时不要猜测，`event_time_key` 留空并将 `time_confidence` 设为 `unknown`。不要用当前对话时间、当前系统时间或 LLM 提炼时间补造事件时间。
- 如果同一 fact 包含多个时间不同的事件，按语义拆分 facts，避免用一个事件时间掩盖互不相关的事件。
- 如果未来问题的答案依赖事件先后、间隔、第一次/上一次或“哪个更早”，fact text 必须同时包含事件对象和时间锚/顺序词，不能只存主题名。
- 如果同一批证据中多个事件可能被未来问题比较先后，应在一条 narrative fact 中明确写出相对顺序，或拆成多条各自带完整背景和时间锚的 facts；不要只保留比较中的一方。
- 带时间锚或顺序词的个人经历即使只是顺带提到，也应认真保留，例如购买、保养/维修、修理、预约、参加活动、旅行、会议、测试、失败、决定等。

提取规则：
1. 提取 0-5 条 facts，但 0 条是常见且正确的结果；不要为了覆盖每一轮、维持话题连续性或解释助手回复而生成 fact。
2. 每条 fact 必须是一段完整叙事，至少包含“议题背景 + 用户已明确表达或确认的关键事实 + 结果/决定/约束/下一步”中的必要要素。助手的观点或动作只可作为经用户确认的结果背景，不能成为叙事核心。
3. 保留可被直接问到且有长期或近期复用价值的具体细节：人名、地点、日期、相对时间、数量、产品、机构、用户明确的约束、决定、计划和偏好。不要仅因助手提到某个细节就保留它。
4. 压缩助手的解释、推导和泛化建议。只有用户明确接受、拒绝、选择、执行，或它已成为后续讨论的具体约束时，才保留相关方案及用户态度。
5. 不要丢弃 “by the way / I also / I just / last Saturday / two months ago / 顺便 / 我还” 这类附带提到、但语义完整的个人事件；如果它们只是模糊片段、无对象的感叹或同一 exchange 的无关上下文，则丢弃。
6. 只有真正互不相关且各自通过写入资格门槛的事件才拆开；时间推理需要比较先后/间隔的事件可以拆成多条，但每条仍必须保留完整背景和时间锚点。
7. 只使用输入证据，不要编造完成状态、意图、原因或用户属性；尤其不要把助手声称的用户爱好、性格、经历或偏好当作用户事实，除非当前批次中用户明确确认。
8. priority 为 0-100。仅输出 priority >= 80 的 fact：90-100 用于稳定身份/偏好/约束、明确决定或重要计划；80-89 用于带明确对象的近期事件、有效计划、用户确认的结果或风险；低于 80 直接丢弃，不要输出。
9. fact_type 只能是 preference、decision、request、recommendation、action、commitment、open_question、risk、error、context、instruction、other；不要仅因助手未回答或用户表述模糊而使用 `open_question`。
10. keywords 只能包含用于检索的短实体、主题、症状、方案、约束、决定和关键时间/顺序锚，通常每个关键词 2-8 个汉字或一个短英文短语；对带时间锚的事件，必须加入原始或补全后的时间词，例如“March 15 2023”“first service”“3/22”“last Saturday”“two months ago”“上周六”“两个月前”。不要把完整句子、寒暄、礼貌话、语气词、泛化表达或“希望这个方法能帮到您”这类文本放入 keywords。
11. 只返回 JSON，不要 markdown。

entity_claim_signal 输出规则：
- `entity_claim_signal` 是当前 fact 对个人世界模型中 entity claim 的结构化证据提示，不是最终 claim，也不能直接决定与已有 claim 的关系。
- `signal_kind` 只能是：`explicit_assertion`（实体明确陈述或可靠记录直接确认的稳定主张）、`pattern_observation`（可在未来与其他 episode 共同支持一个方向明确的规律的观察）。
- `claim_type_hint` 只能是 identity_profile、affiliation、relationship、preference、constraint、behavior_pattern。单次行为不得作为 explicit_assertion 直接生成 behavior_pattern；它至多是 pattern_observation。
- `claim_anchor` 是简短、稳定的聚合标签，用于把不同 episode 中可能描述同一主张或规律的 facts 收拢，例如“安静旅行偏好”“游泳活动习惯”。它不是完整句子，也不是最终 entity claim。
- 只有当前 fact 对某个实体主张或未来规律归纳有实际证据价值时才输出。普通一次性背景、临时建议、助手猜测、寒暄和低价值信息输出空数组。
- 每条 fact 最多输出 3 个 signal。每个 signal 必须包含 entity、signal_kind、claim_type_hint、claim_anchor、evidence_basis、confidence；其中 evidence_basis 必须引用当前 fact 的具体证据，不要引用历史 claim。

future_commitment_signals 输出规则：
- `future_commitment_signals` 是当前 fact 对用户未来世界的证据提示，不是最终 goal、plan 或 work_item，也不能直接决定创建、更新或覆盖已有对象。
- 仅在当前 fact 提供用户未来导向的目标、安排、责任，或这些对象的完成、取消、改期、阻塞等生命周期变化的直接证据时输出；其余情况输出空数组。
- `evidence_kind` 只能是 `goal`、`plan`、`responsibility`、`lifecycle_update`。`candidate_object_types` 只能包含 goal、plan、work_item；goal 只允许来自用户本人明确表达的持续性目标。
- `user_role` 只能是 owner、participant、responsible，表示用户分别是目标拥有者、安排参与者或责任承担者。用户只是被顺带提及时不得输出 signal。
- `assertion_source` 只能是 self_statement、third_party_report、observed_event；`explicitness` 只能是 direct、reported、tentative。助手建议、助手复述、开放假设、条件讨论和推测不得输出 signal。
- `future_commitment_anchor` 是简短、稳定的事件/事项/目标聚合标签，例如“天津客户会议”“半马训练目标”“客户报告交付”；它不是完整句子，也不是最终对象描述。
- 每条 fact 最多输出 2 个 signal。每个 signal 必须包含 subject_entity、evidence_kind、candidate_object_types、user_role、future_commitment_anchor、assertion_source、explicitness、evidence_basis、confidence；其中 evidence_basis 必须是当前 fact 中可直接定位的内容。

输出格式：
{
  "facts": [
    {
      "text": "覆盖完整 exchange 的自包含 narrative fact；优先写清 what，when/where/who/why 仅在证据明确且有助于理解时写入，缺失信息直接省略",
      "keywords": ["关键词1", "关键词2"],
      "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_entity": {"name": "这条 fact 主要描述、影响或归属的单一实体", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
      "fact_root_topic": "稳定的产品/项目/长期议题根主题",
      "fact_aspect_topic": "当前 fact 讨论的具体方面",
      "fact_type": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "event_time_key": "根据对话时间锚点和 fact 内容推导出的事件实际发生时间或代表性时间锚点；无法判断时为空字符串",
      "time_confidence": "explicit|inferred_from_turn|unknown；分别表示原文明确给出、结合当前片段 Time 和相对表达推断、无法判断",
      "where": "明确出现的地点、场景、平台或项目范围；没有明确证据时保持为空字符串，不要填写‘未提及’或类似说明",
      "entity_claim_signal": [
        {
          "entity": {"name": "明确受影响的实体", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
          "signal_kind": "explicit_assertion|pattern_observation",
          "claim_type_hint": "identity_profile|affiliation|relationship|preference|constraint|behavior_pattern",
          "claim_anchor": "用于跨 fact 聚合的具体主张或规律标签",
          "evidence_basis": "当前 fact 中支持该 signal 的具体证据",
          "confidence": 0.8
        }
      ],
      "future_commitment_signals": [
        {
          "subject_entity": "与用户未来相关的主体实体；用户本人填写用户",
          "evidence_kind": "goal|plan|responsibility|lifecycle_update",
          "candidate_object_types": ["goal|plan|work_item"],
          "user_role": "owner|participant|responsible",
          "future_commitment_anchor": "用于跨 fact 聚合的目标、安排或事项标签",
          "assertion_source": "self_statement|third_party_report|observed_event",
          "explicitness": "direct|reported|tentative",
          "evidence_basis": "当前 fact 中支持该 signal 的具体证据",
          "confidence": 0.8
        }
      ]
    }
  ]
}

已有 memory_topic_items 命名候选：
{existing_memory_topic_items}

对话/转写证据批次：
{dialogue_batch}
"""

INTENT_EXTRACTION_PROMPT_ZH = """你负责从已落库、可追溯的 narrative facts 中提取个人世界模型的 Intent & Execution 对象。

只允许输出三类对象：
- goal：主体明确、跨越多个动作的持续性期望结果；一次性愿望或单次动作不是 goal。
- plan：明确的未来安排、行程、会议、活动或事件；必须有时间/时间窗口，或“已约定、已报名、已安排”等安排性证据；它不是待办。
- work_item：责任人明确的有界事项；除具体动作外，必须有交付物、截止时间、明确承诺或明确分配中的至少一项完成锚点。它可以是 personal_action、commitment、assigned 或 external_commitment。

严格限制：
- 只能使用输入 facts 的直接证据。assistant 的建议、推测、行为规律和 prediction 不得自动落库。
- “也许”“如果”“要不要”“考虑一下”等弱假设默认不输出对象。
- goal 或 plan 不得自动拆出 work_item；只有文本明确表达下一步、交付、承诺或分配时才输出 work_item。
- 计划发生是 `occurred`，事项完成是 `completed`，二者不能混淆。
- 每个列表只输出其定义对应的对象；同一 fact 不得自动拆成多类对象，除非 fact 明确提供每一类对象的独立证据。
- `lifecycle_evidence` 只描述当前 fact 可直接观察到的生命周期证据，不是数据库操作；最终 create、update、complete 等操作由后续 reconciliation 结合已有对象决定。

world owner：{world_owner_name}
当前时间：{reference_timestamp}
输入 facts：
{facts}

输出 JSON：
{
  "goal_candidates": [
    {
      "summary": "完整、可展示的持续性目标描述",
      "canonical_key": "用于同义目标匹配的简短稳定名称",
      "owner_entity": "goal 归属实体；没有明确证据时填 world owner",
      "desired_outcome": "目标达成后的结果",
      "success_criteria": "可判定的达成条件；证据不足为空",
      "target_at": "目标时间；证据不足为空",
      "lifecycle_evidence": "none|confirmed|completed|cancelled|blocked",
      "confidence": 0.0,
      "evidence_fact_ids": [1]
    }
  ],
  "plan_candidates": [
    {
      "summary": "完整、可展示的未来安排描述",
      "canonical_key": "用于同义安排匹配的简短稳定名称",
      "actor_entity": "plan 的行动/参与主体",
      "event_or_activity": "未来事件或活动",
      "scheduled_start_at": "计划开始时间，保留已解析时间或原始明确时间",
      "scheduled_end_at": "计划结束时间",
      "time_precision": "exact|day|week|relative|unknown",
      "location": "计划地点",
      "participants": ["其他参与者"],
      "lifecycle_evidence": "none|confirmed|occurred|cancelled|rescheduled|blocked",
      "confidence": 0.0,
      "evidence_fact_ids": [1]
    }
  ],
  "work_item_candidates": [
    {
      "summary": "完整、可展示的责任事项描述",
      "canonical_key": "用于同义事项匹配的简短稳定名称",
      "responsible_entity": "work_item 负责人",
      "beneficiary_entities": ["交付/受益对象"],
      "delegator_entities": ["分配或委托对象"],
      "collaborator_entities": ["协作者"],
      "responsibility_type": "personal_action|commitment|assigned|external_commitment",
      "action_text": "具体动作",
      "deliverable": "交付物或完成结果",
      "due_at": "截止时间",
      "available_from": "可开始执行的时间；证据不足为空",
      "priority": "仅文本明确表达时填写",
      "lifecycle_evidence": "none|confirmed|completed|cancelled|blocked",
      "confidence": 0.0,
      "evidence_fact_ids": [1]
    }
  ]
}

每个 candidate 必须至少引用一个输入 fact id。三个候选列表都必须存在；没有合格对象时对应输出空数组。只返回 JSON。"""


INTENT_RECONCILIATION_PROMPT_ZH = """你是个人世界模型的 Intent & Execution reconciliation 模块。你只判断新的 candidate 与已有对象是否为同一个现实中的 goal、plan 或 work_item，并在确认同一对象后给出其生命周期更新操作。

输入含有：
- `candidates`：由新 facts 提炼出的候选对象。`lifecycle_evidence` 仅表示新 fact 直接观察到的状态证据，不是数据库操作，也不保证已经能定位到已有对象。
- `existing_objects`：同一 world owner 下、当前仍可更新的已有对象。候选与目标对象必须是同一种 `object_type`。

你的工作顺序必须是：
1. 先判断 candidate 是否与某个同类 existing object 是同一个对象；
2. 只有确认同一对象后，才根据 candidate 的内容和 `lifecycle_evidence` 决定 operation；
3. 没有可靠目标对象时输出 `create`，且 `target_object_type` 置为空字符串、`target_object_id` 置为 0。

同一对象的判断规则：
- 不要因为主题、地点、人物或关键词相近就合并。`canonical_key` 只能作为辅助线索，不能单独决定同一性。
- goal：owner 必须相容，且 desired_outcome 指向同一持续性结果。目标时间、成功条件的补充或调整通常是 update，不代表新 goal。
- plan：actor 与核心 event_or_activity 必须相容。相同活动的时间变化是 reschedule；仅地点、参与者或细节补充通常是 update。不同日期的相似活动不自动等同，除非新证据明确说明它是原安排的改期。
- work_item：responsible_entity 与核心 action_text / deliverable 必须相容。截止时间、协作者或交付细节变化通常是 update；不同交付物或不同责任通常是新事项。
- 不得跨类型关联：goal 不能匹配 plan 或 work_item，plan 不能匹配 work_item。

operation 选择规则：
- `create`：没有可靠的同一 existing object。
- `confirm`：同一对象得到再次明确确认，但核心字段和生命周期没有实质变化。
- `update`：同一对象的描述、参与者、地点、目标时间、截止时间、交付物或其他非终态细节得到补充或改变。
- `complete`：仅当新证据明确表示 goal 已实现、plan 已发生（`lifecycle_evidence=occurred`）或 work_item 已完成（`lifecycle_evidence=completed`）时使用。
- `cancel`：仅当新证据明确表示对象被取消、放弃或不再执行时使用。
- `reschedule`：仅用于同一 plan 的明确改期；work_item 截止时间变化使用 update。
- `block`：仅当新证据明确表示该对象当前受阻、无法推进时使用。
- 当 `lifecycle_evidence=none` 时，不要凭空输出 complete、cancel、reschedule 或 block。

每个 candidate_index 必须恰好输出一条 decision，按输入 candidates 的索引编号。reason 用一句简短、可追溯的中文说明同一性或 create 的依据；不要生成输入中不存在的新事实、对象或字段。

只返回 JSON：
{
  "decisions": [
    {
      "candidate_index": 0,
      "operation": "create|confirm|update|complete|cancel|reschedule|block",
      "target_object_type": "goal|plan|work_item",
      "target_object_id": 0,
      "reason": "简短的证据性理由"
    }
  ]
}

candidates：
{candidates}

existing objects：
{existing_objects}
"""


EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH = """你是个人世界模型的显式主张（explicit claim）更新模块。

输入是已经落库、可追溯的 narrative facts。每条 fact 可能带有 `entity_claim_signal`，它只是当前 fact 的结构化提示；必须以 fact summary 与 evidence_fact_ids 为准，不得把 signal 当作额外事实。请只提取 fact 中被说话者明确陈述、或由可靠事件记录直接确认的原子主张；不要把单次行为、建议、助手推测或常识扩展成偏好、习惯或人格结论。

允许的 claim_type 只有：
- identity_profile：身份、背景、角色、稳定画像；
- affiliation：所属组织、团队、项目、地点；
- relationship：人与人/组织/项目的明确关系；
- preference：明确喜欢、厌恶、优先选择；
- constraint：明确限制、禁止、能力边界、持续条件；
- behavior_pattern：本 prompt 禁止输出，行为规律只能由 induction 产生。

规则：
1. subject_entity、object_entity（如有）必须来自输入 facts 的 entities；不能凭空造实体。
2. evidence_fact_ids 必须只引用输入 fact_id。每条 claim 至少引用一个直接支持它的 fact。
3. predicate 使用简短、稳定的小写英文键，例如 has_role、works_with、member_of、located_in、prefers、dislikes、requires、cannot、has_constraint。
4. 关系型主张使用 object_entity；非关系主张的属性、值和限定条件必须完整写入 claim_text。
5. claim_text 是给人阅读的完整、自包含陈述，必须明确主语、关系/属性和值，例如“用户明确偏好安静、节奏较慢的旅行方式”。
6. 否定语义必须写入 predicate 与 claim_text，例如 dislikes、cannot、is_not_member_of；不要输出独立的正负字段。没有直接明示时不要输出。
7. 不要输出一次行程、临时任务、单次建议、待办、开放问题；它们属于 fact / intent 层，而不是 claim。
8. 输出空数组是正确结果。只返回 JSON。

输出：
{
  "claims": [
    {
      "subject_entity": "",
      "claim_type": "identity_profile|affiliation|relationship|preference|constraint",
      "predicate": "",
      "object_entity": "",
      "claim_text": "包含主体和完整语义的陈述",
      "valid_from": "",
      "valid_to": "",
      "evidence_fact_ids": [1],
      "confidence": 0.85
    }
  ]
}

facts：
{facts}
"""


INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH = """你是个人世界模型的规律归纳（inductive claim）模块。

输入是同一实体、同一 claim_anchor 下、来自已完成 episode 的可追溯 facts。claim_anchor 只用于召集候选证据，不是结论；你的任务是保守地判断这些证据是否支持一条规律，而不是重新复述 facts。

只允许输出：preference 或 behavior_pattern，claim_origin 固定由系统写为 inductive。

硬规则：
1. 一条 claim 至少需要 3 个不同的 support_fact_ids，且每个 ID 都必须直接支持当前规律。
2. 单次表达、一次事件、计划、任务、助手建议不能归纳为规律。
3. 不要把偶然行为升级为偏好。只引用直接支持当前规律的 support_fact_ids；其他事实不需要标注为反例。
4. subject_entity 必须是输入给定实体。predicate 使用 prefers、dislikes、usually_does、avoids、has_routine 之一。
5. claim_text 必须是一条带实体主体、属性和值的完整规律陈述；condition_text 为空或简短条件。
6. evidence IDs 必须来自输入。输出空数组是正确结果。只返回 JSON。

输出：
{
  "claims": [
    {
      "subject_entity": "",
      "claim_type": "preference|behavior_pattern",
      "predicate": "prefers|dislikes|usually_does|avoids|has_routine",
      "claim_text": "包含主体和完整规律语义的陈述",
      "condition_text": "",
      "support_fact_ids": [1, 2, 3],
      "confidence": 0.8
    }
  ]
}

候选实体与 claim 聚合线索：
{induction_target}

episode evidence facts：
{facts}
"""


DERIVED_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH = """你是个人世界模型中的直接推导（derived claim）模块。

输入分为两部分：`changed_explicit_claims` 是当前 subject 在本轮新接收事实后发生变化的 explicit claims；`related_active_explicit_claims` 是从数据库读取的、与该 subject 或其直接关联实体相关的历史 active explicit claims。你的任务是仅根据这两部分 claims 的文本和结构，找出能够被直接逻辑推出的新结论。

这里的 derived claim 不是规律归纳、常识补全或可能性猜测。它必须可以写成“因为 premise A（以及 premise B），所以 conclusion C”。例如：
- “张三 reports_to 李四”可以推出“李四 manages 张三”；
- “小王 member_of 团队 Alpha”与“团队 Alpha affiliated_with 公司 X”可以谨慎推出“小王 affiliated_with 公司 X”。

硬规则：
1. 只能基于输入 premise_claim_ids 推导；不得使用外部常识、未给出的背景或自由猜测。
2. 每个 candidate 的 premise_claim_ids 至少包含一个 `changed_explicit_claims` 中的 claim id，且所有 ID 都必须来自输入。
3. 只允许 claim_type 为 affiliation、relationship 或 constraint；禁止输出 identity_profile、preference、behavior_pattern、目标、计划、待办、人格或风险判断。
4. subject_entity_id 和 object_entity_id（0 表示无 object）必须来自输入中出现的实体 ID；不能创建新实体。
5. predicate 使用简短、稳定的小写英文键；claim_text 必须是完整、自包含、面向阅读者的结论。
6. 不要重复或改写某个 premise 本身；没有严格成立的新结论时返回空数组。
7. confidence 表示“在给定 premise 下该结论成立”的把握，不表示 premise 本身的真实性。只返回 JSON。

输出：
{
  "claims": [
    {
      "subject_entity_id": 0,
      "claim_type": "affiliation|relationship|constraint",
      "predicate": "",
      "object_entity_id": 0,
      "claim_text": "完整、自包含的推导结论",
      "premise_claim_ids": [1, 2],
      "confidence": 0.8
    }
  ]
}

changed explicit claims：
{changed_claims}

related active explicit claims：
{related_claims}
"""


ENTITY_CLAIM_RECONCILIATION_PROMPT_ZH = """你是个人世界模型中的 entity claim reconciliation 模块。

输入只包含新旧 entity claim 的文本。你的任务只通过这些文本的自然语言语义，判断每个 candidate 与已有 claim 的关系；不要推断来源可信度、不要考虑 claim_origin、置信度、时间、数据库状态或后续写入策略。

semantic_relation 只能是：
- duplicate：两条文本表达同一主张；
- supports：candidate 是旧 claim 的直接支持或重复佐证，但表达不完全相同；
- contradicts：两条文本不能同时为真，但文本本身没有明确的替换/更新含义；
- refines：candidate 为旧 claim 增加条件、范围、例外或更具体表述；
- supersedes：candidate 文本明确表示旧主张被改变、否定、修正或替代；
- unrelated：两条文本可以同时成立，或没有明确关系。

不同偏好、角色或关系不天然冲突，例如“喜欢绘画”和“喜欢钢琴”通常是 unrelated。一个 candidate 可以同时与多个已有 claim 存在关系；请列出所有明确相关的 existing claim。没有相关 claim 时 relations 输出空数组。只返回 JSON。

输出：
{
  "decisions": [
    {
      "candidate_claim_index": 0,
      "relations": [
        {
          "existing_claim_id": 12,
          "semantic_relation": "duplicate|supports|contradicts|refines|supersedes",
          "confidence": 0.9,
          "reason": "简短说明"
        }
      ]
    }
  ]
}

candidate_claims：
{candidate_claims}

existing_claims：
{existing_claims}
"""


RECALL_QUERY_ANALYSIS_PROMPT_ZH = """你是 AI 眼镜长期记忆系统中的 recall query 分析器。

请先理解当前记忆结构，再分析用户查询的检索方式。

记忆结构：
1. `fact`：从一次 episode 的对话或全天候转写中提炼出的、可追溯且自包含的 narrative evidence。它保留具体发生了什么、谁参与、时间、地点/场景、原因、观点变化、建议、接受/拒绝、约束、结论和未解决问题等证据。fact 通常带有 `summary`、`keywords`、`entities`、`fact_root_topic`、`fact_aspect_topic`、`event_time_key` 和 `dialogue_time_key`。
2. `entity_claim`：从多个 facts 反思得到、具有状态和置信度且可回溯支撑 facts 的实体主张，例如稳定偏好、习惯、关系、背景、约束或特征。
3. `goal`、`plan`、`work_item`：从 facts 中提炼出的未来导向对象，分别表示期望结果、明确安排和可闭环责任事项；其 status 和目标/开始/截止时间具有重要语义。
4. `episode`：连续原始片段的经历摘要，只用于在 facts 之间建立关联，绝不作为 direct recall object。

判断准则：
- 只有当 query 明确指向用户与助手的主动对话，才偏向 `assistant_wakeup`；明确指向全天录音、会议、旁听、多人数对话，才偏向 `allday_recording`；不确定时两者都保留。
- `recall_object_types` 是本次可直接检索的对象类型。必须始终包含 `fact`；仅在 query 明确需要稳定实体知识时加入 `entity_claim`；仅在 query 询问未来目标、安排、责任、截止、未完成事项或其状态变化时加入相应的 `goal`、`plan`、`work_item`。绝不能输出 `episode`。
- 具体发生了什么、日期、地点、人名、原话语义、事件先后和可追溯证据，优先 `fact`。
- 稳定偏好、长期约束、习惯/流程、关系画像或个人背景，除 `fact` 外可加入 `entity_claim`；仍需保留 `fact` 以提供证据。
- 任务、承诺、未来安排、目标、截止与未完成事项，除 `fact` 外可加入对应 prospective object；仍需保留 `fact` 以提供证据。
- 不确定时保持宽检索，漏掉证据比少取几个候选更糟。
- `keywords` 输出 2-8 个短检索词，优先保留具体人物、组织、产品、项目、主题、动作、结果和约束；不要输出完整句子、寒暄、泛化词或普通时间表达。
- `entities` 输出对语义检索有帮助的实体名称及类型。实体可以是人物、组织、地点、产品、项目、技术或具体概念；普通的“今天/昨天/上周”等时间表达不要作为实体。
- `temporal_mode` 表示时间范围应该匹配哪一种 fact 时间：`event_time` 表示事实描述的现实事件时间，`dialogue_time` 表示对话/转写发生时间，`both` 表示任一时间命中即可，`none` 表示不做时间硬过滤。询问“做了什么/发生了什么/买过什么”优先使用 `event_time`；询问“讨论了什么/提到过什么/问过什么”优先使用 `dialogue_time`；无法判断时使用 `none`。
- `temporal_bounds` 由你根据原始 query 和参考时间解析；`start` / `end` 使用 `YYYY-MM-DD HH:MM:SS` 或 `null`，至少提供一个边界。`end` 是排他上界。没有时间约束时输出 `null`。
- `is_prospective_time_slot_query` 仅在用户询问自己在当前或未来某个时间窗口内“有什么安排/计划/待办/未完成事项”这类集合或状态时为 true。出行建议、天气新闻等外部信息、单一事实确认、纯预测，以及“提醒我/帮我创建日程”等即时指令均为 false。

只返回 JSON：
{
  "recall_object_types": ["fact"],
  "needs_broad_evidence": false,
  "query_rewrite": "面向统一记忆检索的改写",
  "keywords": ["关键词1", "关键词2"],
  "entities": [{"name": "实体名", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}],
  "temporal_bounds": {"start": "YYYY-MM-DD HH:MM:SS|null", "end": "YYYY-MM-DD HH:MM:SS|null"},
  "temporal_mode": "event_time|dialogue_time|both|none",
  "is_prospective_time_slot_query": false
}

原始用户查询：
{query}

解析相对时间表达时使用的参考时间：
{reference_time}
"""
