"""English prompt templates for the unified memory prototype."""

MEMORY_RETRIEVED_FORMAT_PROMPT_EN = """[Unified Memory]
System note: Recall may contain traceable fact evidence, entity claims, and active goals, plans, or work items. Claims and prospective objects are derived memories supported by facts; do not present them as unsupported new facts.
System note: For facts, dialogue_time is when the conversation/transcript discussed the fact, while event_time is when the real-world event described by the fact occurred. They are different fields; an unknown event_time must not be inferred from dialogue_time.
{memory_sections}"""

MEMORY_RETRIEVED_SECTION_SPECS_EN = (
    (
        "[Retrieved Facts]",
        "These are ranked narrative facts retrieved directly from memory_facts.",
        "fact",
    ),
    (
        "[Entity Knowledge]",
        "These are active entity claims. Interpret them with their supporting facts and status.",
        "entity_claim",
    ),
    (
        "[Active Goals]",
        "These are active goals; their current status and target time take precedence over older related facts.",
        "goal",
    ),
    (
        "[Active Plans]",
        "These are active plans. A plan must not be treated as already completed.",
        "plan",
    ),
    (
        "[Open Work Items]",
        "These are not-closed work items. Respect their responsible party, due time, and status.",
        "work_item",
    ),
)

ENTITY_EXTRACTION_GUIDANCE_EN = """Entity extraction rules:

Entities are not limited to traditional named entities. In this memory system, an entity is a semantic anchor that can be reused across facts for clustering, retrieval, and graph construction.
Prefer nouns or short noun phrases with long-term memory value instead of only proper names.

Allowed entity type values:
- PERSON: specific people, names, roles, or speakers mentioned in the conversation
- ORGANIZATION: companies, teams, institutions, or groups
- LOCATION: geographic locations or venues
- PRODUCT: products or services
- PROJECT: projects, product names, or long-running work items
- TECHNOLOGY: technology stacks, frameworks, libraries, tools, APIs, or systems
- CONCEPT: abstract concepts, methods, theories, or reusable ideas
- TOPIC: discussed domains or subject areas
- PREFERENCE: user preferences, likes, habits, or dislikes
- OTHER: explicitly mentioned entities that do not fit the above types

Entities that should be extracted include:
- Conversation subjects or roles, such as user, assistant, speaker_1, speaker_2, spouse, child, team, or client
- User-relevant domains, problems, tasks, states, or scenarios, such as health management, physical condition, work, business events, family education, communication, or fatigue
- Reusable plans, methods, tools, activities, or objects, such as healthy eating, family meetings, shared rules, picnics, or yoga mats
- Constraint objects or conditions that affect user choices, such as financial burden, fixed schedule, time shortage, or work pressure

Do not extract ordinary time expressions as entities, such as today, yesterday, last week, the past three days, 2026-05-07, 10:30, or three months.
Time should be stored as fact time metadata, not in the entity graph.
Only named time concepts with semantic identity may be entities, such as Spring Festival, Q3 earnings season, or Sprint 42.

Do not extract pure attributes, adjective-only phrases, isolated degree words, or generic labels as entities; keep them in fact text, keywords, topics, or states instead.
For example: low venue dependence, low intensity, high priority, low cost, strong privacy, lightweight.
If a phrase contains a reusable object or scenario, extract the core object or scenario:
- "fatigue caused by long-term high-intensity work" may yield "work" and "fatigue"
- "high-frequency business activities" may yield "business activities"
- "the financial burden is too heavy" may yield "financial burden"

Entities must come from explicit conversation content or be directly determined by a role/fact subject. Do not over-infer.
Each retained memory fact should usually include a subject entity, such as user/assistant/speaker, plus 1-4 core semantic anchors."""


EPISODE_SUMMARY_PROMPT_EN = """You are the episode aggregation module for a long-term memory system. The input contains raw interaction/transcript segments from one continuous interval and high-value facts extracted from the same evidence. Generate a narrative summary of one reviewable experience.

Input roles:
- `source_segments` are chronological source evidence and the sole primary basis for the episode's content, ordering, and stance.
- `evidence_facts` are durable coverage anchors selected from those segments. Use them only to check that important information is not missed; never rewrite, concatenate, or expand them one by one.

Conceptual boundaries:
- A fact is an independently retrievable atomic evidence unit containing precise details.
- An episode is a higher-level narrative of what happened during one continuous experience: what unfolded, what it centered on, and what conclusion or open point remained.
- An episode is not a long-term state, user profile, cross-experience pattern, risk assessment, or actionable item. Never generalize one episode into a durable conclusion.

Generation rules:
1. Reconstruct the experience from source segments: context/trigger -> discussion, observation, or action -> meaningful turn or stance -> result, decision, or unresolved point. Omit any unsupported step.
2. Abstract related expressions, but do not replay every turn or write one sentence per fact. The summary must be more coherent and contextual than the facts, not a longer fact list.
3. Ignore wake words, greetings, courtesy closings, repetitions, empty acknowledgements, and templated assistant language. Retain only the people/objects, setting, issue, user concern, key response, decision, constraint, outcome, or open question needed to understand the experience.
4. Cover high-value information in `evidence_facts` where it agrees with source segments. If facts conflict with, overstate, or leave ambiguity in the source, follow the source and preserve uncertainty.
5. Preserve chronology, conditions, and speaker stance. Do not turn advice into a decision, a possibility or plan into completion, an assistant view into a user belief, or invent causality, owners, deadlines, or outcomes.
6. If the continuous interval contains several unrelated but valuable topics, connect them with a structured summary without forcing a causal link or omitting important content.
7. summary must be one concise, self-contained experience narrative: understandable without source segments and clear about the central people/objects, progression, and conclusion/open point.
8. title is a short, concrete, distinguishable retrieval title. Prefer "object + core event/problem/decision" over broad labels.
9. canonical_topics contains only 1-3 stable topics. Prefer `fact_root_topic` values from evidence facts; never use an action, one-off conclusion, emotion, or isolated keyword as a topic.
10. Return JSON matching the schema exactly. No markdown, explanation, or extra fields.

Output schema:
{
  "title": "short specific title",
  "summary": "self-contained episode summary",
  "canonical_topics": ["stable topic 1", "stable topic 2"]
}

Raw source segments:
{source_segments}

High-value evidence facts (coverage checks only; do not concatenate):
{evidence_facts}
"""


UNIFIED_MEMORY_EXTRACTION_PROMPT_EN = """You are the memory extraction module for a unified AI-glasses memory system inspired by MemPalace.

Current memory structure:
- fact: a traceable, self-contained, independently retrievable narrative evidence unit extracted from a continuous evidence batch. Facts are persisted first and may be assigned to an episode later.
- episode: a higher-level experience summary generated by a separate module from persisted raw interaction/transcript segments in one continuous interval; newly generated facts are coverage anchors only, so it is not a copy of each input batch or fact.
- memory_topic_items: a reusable topic naming registry populated from stored facts and episodes. It contains canonical topics and fact aspects, not a topic summary or state.
- entity_claim: a traceable entity proposition projected from facts by a separate reflection task; it is not direct evidence from the current dialogue.
- goal / plan / work_item: future desired outcomes, explicit arrangements, and closed-loop responsibilities extracted by a separate Intent & Execution task after facts are stored. Do not output them here.

Your task now is to extract Hindsight-style high-quality narrative facts from the chronological dialogue/transcript evidence batch below. Episode summary and episode canonical_topics are generated by a separate module from persisted source segments with extracted facts as coverage anchors; do not output episode-level fields in this prompt.

Topic and entity rules:
- A fact's `fact_root_topic` must be grounded in the main durable issue in current evidence, while `fact_aspect_topic` keeps the concrete aspect under that root. Use a conservative, specific topic when evidence is limited.
- Every fact must also output one `primary_entity`, the single entity that the fact mainly describes, affects, or belongs to; it must be one object, not an array.
- `primary_entity` must come from the fact's `entities`. Do not choose an entity merely because it is mentioned, provides a recommendation, or is a location, tool, or background context. For a multi-person exchange, choose the person or entity mainly described or affected by the fact; for a fact about the user's own preference, habit, constraint, or risk, choose the user.
- Keep `entities` for all directly relevant entities so retrieval preserves participants and context; downstream entity-state matching uses only `primary_entity`, so one fact must not be assigned to multiple entities.

""" + ENTITY_EXTRACTION_GUIDANCE_EN + """

Core Hindsight-style narrative fact requirements:
- Each fact should cover a complete exchange or a clear topic segment, not a single utterance. Do not mechanically split "the user raised a problem", "the assistant suggested a solution", and "the user accepted/rejected it" into separate fragments; if they respond to the same issue, merge them into one narrative fact.
- Each fact must be understandable without reading the original dialogue and preserve the pragmatic flow of the interaction: why the user raised the issue, what the assistant suggested, how the user responded, and what preference, decision, constraint, unresolved question, or next step emerged.
- Each fact must naturally include the five dimensions in its text: what (complete event/topic/plan/conclusion), when (conversation timestamp or explicit time anchor), where (location/setting/platform/project scope; if absent, say no specific location/setting was mentioned), who (user, assistant, and other key people/organizations with their roles), and why (explicit reason, motivation, concern, disagreement, constraint, implication, conclusion, or follow-up).
- For a roughly five-turn dialogue batch or a coherent multi-speaker transcript segment, usually produce 1-3 facts. Only split when the batch truly contains multiple unrelated events/topics. In most cases, do not exceed 5 facts.

fact_type classification:
- `fact_type` describes the fact's primary semantic role. It must be one of preference, decision, request, recommendation, action, commitment, open_question, risk, error, context, instruction, or other.
- Choose the one category that best represents the fact's memory value. Do not use `open_question` merely because the assistant did not answer or the user's wording is vague.

Temporal fidelity requirements:
- Preserve sequence and ordering expressions exactly when they affect meaning: first, first time, second, previous, next, later, earlier, before, after, once, again, subsequent, prior, last, most recent, and similar wording. Do not paraphrase away order. For example, keep "serviced for the first time on March 15" rather than reducing it to "had a good service experience".
- Preserve relative time expressions in text and keywords: yesterday, last Saturday, previous week, two months ago, about a month ago, mid-February, recently, shortly after, and similar phrases. If the expression can be resolved unambiguously from the Conversation timestamp, write the resolved real-world event time directly into `event_time_key`.
- The `Time` shown for each segment is the dialogue/transcript timestamp. It is only the reference anchor for resolving relative event times, not the default event time for the fact. Derive `event_time_key` from the specific event described by the fact and the temporal evidence in the text; do not copy the dialogue timestamp merely because the event was discussed at that time.
- `event_time_key` is the most representative real-world occurrence time or temporal anchor for the event described by the fact. It is not the dialogue time, extraction time, or current system time. It is a single time field; do not output an event end time, interval, or additional start/end time fields.
- Use this priority when deriving time: explicit absolute date/time in the evidence > a relative expression that can be resolved unambiguously from the segment `Time` > an event explicitly described as happening during the current conversation. For example, with dialogue time 2023-05-30, `last month (around April 2023)` should resolve to a representative time around 2023-04 rather than 2023-05-30; `last weekend (May 27-28)` should use a representative time around 2023-05-27; only an event explicitly described as decided/completed today should use 2023-05-30.
- If a fact describes both a current conversational act and an earlier background event, use the time of the event the fact primarily describes; split the fact when necessary instead of allowing the dialogue timestamp to overwrite the earlier event time. If only a month, weekend, or relative period is supported, preserve the original expression in text/keywords and use a conservative representative time anchor in `event_time_key`.
- Prefer explicit dates, times, weekdays, and relative time expressions from the evidence. Resolve a relative expression to an absolute time only when the current segment `Time` makes the resolution unambiguous; otherwise do not guess, leave `event_time_key` empty, and set `time_confidence` to `unknown`. Never fabricate an event time from the dialogue timestamp, current time, or extraction time.
- If one fact contains multiple events with different times, split them into separate facts instead of using one event time to hide unrelated events.
- If an event's answerability depends on temporal order, the fact text must include both the event object and the time anchor or order marker. Do not store only the topic name.
- If multiple events in the batch may later be compared by before/after/first/which happened earlier, either keep them in one narrative fact that explicitly states their relative order, or split them into separate complete facts with their own time anchors. Avoid keeping only one side of a comparison.
- Personal events mentioned as side context remain important when they include time anchors or ordering words, such as purchases, service/maintenance, repairs, appointments, attendance, travel, meetings, tests, failures, and decisions.

Rules:
1. Extract 0-5 facts. Do not force a fact for every turn.
2. Each fact must be a complete narrative preserving the essential topic background plus the flow of user/assistant views or actions and an explicit reason, disagreement, constraint, conclusion, or next step.
3. Preserve concrete answerable details: names, places, titles, colors, dates, weekdays, relative times, numbers, amounts, durations, products, organizations, recommendations, constraints, decisions, and user preferences.
4. Do not drop personal events mentioned as asides, e.g. "by the way", "I also", "I just", "last Saturday", "two months ago"; but if they are context inside the same exchange, merge them into the same narrative fact instead of emitting context-free short notes.
5. Split only truly unrelated events. Events that must be compared for temporal reasoning may be split, but every split fact must still preserve its own background and time anchor.
6. Use only the dialogue evidence. Do not invent completion, intent, or reasons.
7. Keep assistant recommendations that contain concrete future-answerable items inside the relevant exchange narrative, and include whether the user accepted, rejected, hesitated, or added constraints when supported.
8. priority is 0-100. Keep only facts worth at least 60.
9. fact_type must be preference, decision, request, recommendation, action, commitment, open_question, risk, error, context, instruction, or other.
10. Do not output short facts like "the user said X" or "the assistant suggested Y". If deleting the topic background, reason, disagreement, or conclusion would make the text a vague short note, add those details back; if the dialogue does not support them, omit the fact.
11. Do not store assistant pleasantries, generic closings, or low-information encouragement as standalone facts, e.g. "hope this helps", "let me know if you have other questions", "okay", or "you're welcome", unless they explicitly change a decision, commitment, or next step.
12. keywords must be short retrieval terms: entities, topics, symptoms, plans, constraints, decisions, and important time/order anchors. For time-sensitive facts, include the original or resolved time phrase such as "March 15 2023", "first service", "3/22", "last Saturday", or "two months ago". Do not put full sentences, pleasantries, filler, generic encouragement, or phrases like "hope this method helps you" into keywords.
13. Return JSON only. No markdown.

entity_claim_signal rules:
- `entity_claim_signal` is a structured evidence hint from this fact for entity claims in the personal world model. It is not a final claim and must not decide its relation to existing claims.
- `signal_kind` must be one of: `explicit_assertion` (a durable proposition directly stated by an entity or confirmed by a reliable record), or `pattern_observation` (an observation that may later support one directionally specific pattern together with other episodes).
- `claim_type_hint` must be one of: identity_profile, affiliation, relationship, preference, constraint, behavior_pattern. A single action must not become an explicit_assertion for behavior_pattern; at most it is a pattern_observation.
- `claim_anchor` is a short, stable grouping label that can collect facts from different episodes about one possible claim or pattern, such as "quiet travel preference" or "swimming routine". It is neither a sentence nor the final entity claim.
- Output signals only when this fact has concrete evidence value for an entity claim or later pattern induction. Use an empty array for one-off background, temporary suggestions, assistant speculation, pleasantries, and low-value content.
- Return at most 3 signals per fact. Every signal must contain entity, signal_kind, claim_type_hint, claim_anchor, evidence_basis, and confidence. evidence_basis must cite the current fact, never a prior claim.

prospective_signals rules:
- `prospective_signals` are structured hints that this fact is evidence about the user's future world. They are not final goals, plans, or work items, and must not directly create, update, or override an existing object.
- Output a signal only when this fact directly supports a user's future-facing goal, arrangement, responsibility, or a lifecycle change such as completion, cancellation, rescheduling, or blockage. Otherwise output an empty array.
- `evidence_kind` must be one of goal, plan, responsibility, or lifecycle_update. `candidate_object_types` may contain only goal, plan, and work_item. A goal may be signalled only by the user's own explicit durable goal expression.
- `user_role` must be owner, participant, or responsible: the user respectively owns the goal, participates in the arrangement, or bears the responsibility. Do not output a signal when the user is merely mentioned.
- `assertion_source` must be self_statement, third_party_report, or observed_event; `explicitness` must be direct, reported, or tentative. Assistant suggestions or restatements, open hypotheticals, conditional discussion, and speculation never create signals.
- `prospective_anchor` is a short, stable grouping label for the goal, event, or responsibility, such as "Tianjin client meeting", "half-marathon training goal", or "client report delivery". It is not a full sentence or the final object description.
- `operation_hint` must be create, confirm, update, complete, cancel, reschedule, or block. Use a non-create value only when the fact directly states that lifecycle change.
- Return at most 2 signals per fact. Every signal must include subject_entity, evidence_kind, candidate_object_types, operation_hint, user_role, prospective_anchor, assertion_source, explicitness, evidence_basis, and confidence. evidence_basis must be concrete evidence in this fact.

Output schema:
{
  "facts": [
    {
      "text": "self-contained narrative fact covering a complete exchange and expressing what/when/where/who/why, with background, user/assistant view or action flow, and reason/disagreement/constraint/conclusion/next step",
      "keywords": ["keyword1", "keyword2"],
      "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"}],
      "primary_entity": {"name": "the single primary entity of this fact", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
      "fact_root_topic": "stable product/project/long-running issue root topic",
      "fact_aspect_topic": "specific aspect discussed by this fact",
      "fact_type": "preference|decision|request|recommendation|action|commitment|open_question|risk|error|context|instruction|other",
      "priority": 80,
      "event_time_key": "real-world event occurrence time or representative temporal anchor derived from the dialogue time anchor and fact content; empty when it cannot be determined",
      "time_confidence": "explicit|inferred_from_turn|unknown; explicit evidence, resolved from the segment Time and a relative expression, or undetermined",
      "where": "",
      "entity_claim_signal": [
        {
          "entity": {"name": "explicitly affected entity", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|TOPIC|PREFERENCE|OTHER"},
          "signal_kind": "explicit_assertion|pattern_observation",
          "claim_type_hint": "identity_profile|affiliation|relationship|preference|constraint|behavior_pattern",
          "claim_anchor": "specific claim or pattern grouping label",
          "evidence_basis": "specific evidence from this fact supporting the signal",
          "confidence": 0.8
        }
      ],
      "prospective_signals": [
        {
          "subject_entity": "the entity whose future is involved; use the world owner for the user",
          "evidence_kind": "goal|plan|responsibility|lifecycle_update",
          "candidate_object_types": ["goal|plan|work_item"],
          "operation_hint": "create|confirm|update|complete|cancel|reschedule|block",
          "user_role": "owner|participant|responsible",
          "prospective_anchor": "a stable label for grouping the goal, arrangement, or responsibility",
          "assertion_source": "self_statement|third_party_report|observed_event",
          "explicitness": "direct|reported|tentative",
          "evidence_basis": "concrete evidence from this fact supporting the signal",
          "confidence": 0.8
        }
      ]
    }
  ]
}

Dialogue/transcript evidence batch:
{dialogue_batch}
"""

INTENT_EXTRACTION_PROMPT_EN = """Extract Intent & Execution objects from stored, traceable narrative facts for a personal world model.

Only output:
- goal: an explicit, durable desired outcome that spans more than one action;
- plan: an explicit future arrangement, event, activity, trip, meeting, or appointment with a time/window or explicit arranged, agreed, or registered evidence;
- work_item: a bounded responsibility with a clear responsible party, action, and at least one completion anchor: a deliverable, deadline, explicit commitment, or explicit assignment.

Use only direct fact evidence. Do not store assistant suggestions, speculation, behavior patterns, or predictions. Do not create an object for weak hypotheticals such as maybe, if, should we, or consider. A goal or plan never automatically creates a work item. Use `occurred` only for a plan whose event happened, and `completed` only for a completed work item. Keep each candidate in its matching list; do not fan out one fact into multiple object types without independent evidence. `lifecycle_evidence` describes only an observed fact-level change, never a database operation; reconciliation decides create, update, or completion against existing objects.

World owner: {world_owner_name}
Reference time: {reference_timestamp}
Facts:
{facts}

Return JSON only:
{
  "goal_candidates": [
    {
      "summary": "complete durable outcome description",
      "canonical_key": "short stable goal identity key",
      "owner_entity": "goal owner",
      "desired_outcome": "outcome after achievement",
      "success_criteria": "checkable success condition or empty",
      "target_at": "goal target time",
      "lifecycle_evidence": "none|confirmed|completed|cancelled|blocked",
      "confidence": 0.0,
      "evidence_fact_ids": [1]
    }
  ],
  "plan_candidates": [
    {
      "summary": "complete future arrangement description",
      "canonical_key": "short stable plan identity key",
      "actor_entity": "plan actor",
      "event_or_activity": "future event or activity",
      "scheduled_start_at": "plan start time",
      "scheduled_end_at": "plan end time",
      "time_precision": "exact|day|week|relative|unknown",
      "location": "plan location",
      "participants": ["other plan participants"],
      "lifecycle_evidence": "none|confirmed|occurred|cancelled|rescheduled|blocked",
      "confidence": 0.0,
      "evidence_fact_ids": [1]
    }
  ],
  "work_item_candidates": [
    {
      "summary": "complete responsibility description",
      "canonical_key": "short stable work item identity key",
      "responsible_entity": "work item responsible party",
      "beneficiary_entities": ["beneficiaries"],
      "delegator_entities": ["delegators"],
      "collaborator_entities": ["collaborators"],
      "responsibility_type": "personal_action|commitment|assigned|external_commitment",
      "action_text": "work item action",
      "deliverable": "work item deliverable",
      "due_at": "work item due time",
      "available_from": "earliest execution time or empty",
      "priority": "only when explicit",
      "lifecycle_evidence": "none|confirmed|completed|cancelled|blocked",
      "confidence": 0.0,
      "evidence_fact_ids": [1]
    }
  ]
}
Every candidate needs at least one input fact id. All three candidate lists must be present; use an empty list when none qualify."""


INTENT_RECONCILIATION_PROMPT_EN = """Decide only the relationship between new Intent & Execution candidates and existing objects. Do not invent facts or modify object fields.

For each candidate, choose a matching existing object only when it represents the same goal, plan, or work item with compatible subject, core activity/deliverable, and time. Return create when no reliable match exists. Plan `occurred` means the event happened; work_item `completed` means the responsibility was fulfilled.

Return JSON only:
{
  "decisions": [
    {
      "candidate_index": 0,
      "operation": "create|confirm|update|complete|cancel|reschedule|block",
      "target_object_type": "goal|plan|work_item",
      "target_object_id": 0,
      "reason": "brief evidence-grounded reason"
    }
  ]
}

candidates:
{candidates}

existing objects:
{existing_objects}
"""


EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_EN = """You update explicit claims in a personal world model.

The input contains stored, traceable narrative facts. Each fact may include an `entity_claim_signal`, which is only a structured hint from that fact; use the fact summary and evidence_fact_ids as the evidence, never the signal as an additional fact. Extract only atomic claims explicitly stated by a speaker or directly confirmed by a reliable event record. Do not infer a preference, habit, or personality conclusion from a one-off action, a recommendation, or assistant speculation.

Allowed claim_type values: identity_profile, affiliation, relationship, preference, constraint. Never output behavior_pattern here; it is induction-only.

Rules:
1. subject_entity and object_entity, if present, must occur in input fact entities.
2. evidence_fact_ids must cite only input fact IDs and each claim needs direct evidence.
3. Use concise stable lowercase predicates, such as has_role, works_with, member_of, located_in, prefers, dislikes, requires, cannot, has_constraint.
4. Use object_entity for relational claims. For non-relational claims, express the attribute, value, and relevant qualification completely in claim_text.
5. claim_text is a complete, self-contained, reader-facing proposition with its subject, predicate, and value.
6. Encode negation in predicate and claim_text, such as dislikes, cannot, or is_not_member_of; do not output a standalone positive/negative field. Return no claim without direct evidence.
7. Do not output a one-time trip, task, plan, recommendation, or open question. An empty list is correct. Return JSON only.

Output:
{
  "claims": [{
    "subject_entity": "",
    "claim_type": "identity_profile|affiliation|relationship|preference|constraint",
    "predicate": "",
    "object_entity": "",
    "claim_text": "complete proposition with subject and meaning",
    "valid_from": "",
    "valid_to": "",
    "evidence_fact_ids": [1],
    "confidence": 0.85
  }]
}

facts:
{facts}
"""


INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_EN = """You conservatively consolidate inductive claims for a personal world model.

Input facts are traceable observations for one entity and one claim_anchor, drawn from completed episodes. The claim_anchor only gathers candidate evidence; it is not a conclusion. Decide whether the evidence supports a pattern; do not summarize facts.

Only preference and behavior_pattern claims are allowed.

Hard rules:
1. A claim requires at least three distinct support_fact_ids, and every cited ID must directly support the proposed pattern.
2. A single statement, event, plan, task, or assistant recommendation is not a pattern.
3. Do not turn an isolated behavior into a preference. Cite only facts that directly support the proposed pattern in support_fact_ids; other facts need not be labeled as counterexamples.
4. subject_entity must equal the given subject. predicate is one of prefers, dislikes, usually_does, avoids, has_routine.
5. claim_text is a complete reader-facing pattern proposition with its subject, property, and value.
6. Evidence IDs must be input IDs. An empty list is correct. Return JSON only.

Output:
{
  "claims": [{
    "subject_entity": "",
    "claim_type": "preference|behavior_pattern",
    "predicate": "prefers|dislikes|usually_does|avoids|has_routine",
    "claim_text": "complete pattern proposition with subject and meaning",
    "condition_text": "",
    "support_fact_ids": [1, 2, 3],
    "confidence": 0.8
  }]
}

candidate entity and claim grouping hint:
{induction_target}

episode evidence facts:
{facts}
"""


DERIVED_ENTITY_CLAIM_EXTRACTION_PROMPT_EN = """You produce direct derived claims for a personal world model.

The input has two sections. `changed_explicit_claims` are explicit claims for the current subject that changed after this incoming fact batch. `related_active_explicit_claims` are historical active explicit claims loaded from the database because they concern this subject or its directly linked entities. Produce only conclusions that follow directly and necessarily from those claims.

A derived claim is not pattern induction, common-sense completion, or a plausible guess. It must be explainable as “because premise A (and premise B), conclusion C.” For example, “Alice reports_to Bob” can imply “Bob manages Alice”; “Alice member_of Team Alpha” plus “Team Alpha affiliated_with Company X” can cautiously imply “Alice affiliated_with Company X.”

Rules:
1. Use only the supplied premise_claim_ids; do not use external knowledge or unstated background.
2. Every candidate must cite at least one premise from `changed_explicit_claims`, and every cited ID must be in the input.
3. Only affiliation, relationship, and constraint claim types are allowed. Never create identity_profile, preference, behavior_pattern, goals, plans, work items, personality labels, or risk judgments.
4. subject_entity_id and object_entity_id (0 means no object) must be entity IDs appearing in the input. Never create an entity.
5. Use concise stable lowercase predicates and a complete, self-contained reader-facing claim_text.
6. Do not repeat or merely paraphrase a premise. Return an empty list when no strictly entailed new conclusion exists.
7. confidence is confidence that the conclusion follows given the premises, not confidence that the premises are true. Return JSON only.

Output:
{
  "claims": [{
    "subject_entity_id": 0,
    "claim_type": "affiliation|relationship|constraint",
    "predicate": "",
    "object_entity_id": 0,
    "claim_text": "complete self-contained derived conclusion",
    "premise_claim_ids": [1, 2],
    "confidence": 0.8
  }]
}

changed explicit claims:
{changed_claims}

related active explicit claims:
{related_claims}
"""


ENTITY_CLAIM_RECONCILIATION_PROMPT_EN = """You reconcile entity claims in a personal world model.

The input contains only the natural-language text of new candidate claims and existing claims. Determine their semantic relationship from text alone. Do not infer source reliability, claim origin, confidence, time, database status, or any write strategy.

semantic_relation is one of:
- duplicate: the texts state the same proposition;
- supports: the candidate directly supports or repeats the existing claim without identical wording;
- contradicts: the texts cannot both be true, but the candidate does not explicitly replace or update the existing claim;
- refines: the candidate adds a condition, scope, exception, or more specific form;
- supersedes: the candidate text explicitly says that the old proposition changed, was negated, corrected, or replaced;
- unrelated: both can be true or no clear relation exists.

Different preferences, roles, or relationships are not inherently contradictory: liking painting and liking piano are normally unrelated. One candidate can relate to multiple existing claims; list every clearly related existing claim. Return an empty relations array when none is related. Return JSON only.

Output:
{
  "decisions": [{
    "candidate_claim_index": 0,
    "relations": [{
      "existing_claim_id": 12,
      "semantic_relation": "duplicate|supports|contradicts|refines|supersedes",
      "confidence": 0.9,
      "reason": "short explanation"
    }]
  }]
}

candidate_claims:
{candidate_claims}

existing_claims:
{existing_claims}
"""


RECALL_QUERY_ANALYSIS_PROMPT_EN = """You are the recall query analyzer for the AI-glasses long-term memory system.

Understand the memory structure before analyzing the query. 

Memory structure:
1. `fact`: traceable, self-contained narrative evidence extracted from one conversation episode or all-day transcript. It preserves what happened, participants, time, place or scene, reasons, viewpoint changes, suggestions, acceptance or rejection, constraints, conclusions, and unresolved questions. Facts usually include `summary`, `keywords`, `entities`, `fact_root_topic`, `fact_aspect_topic`, `event_time_key`, and `dialogue_time_key`.
2. `entity_claim`: a statusful, confidence-weighted entity proposition reflected from facts, such as a stable preference, routine, relationship, background, constraint, or characteristic. It remains traceable to supporting facts.
3. `goal`, `plan`, and `work_item`: future-oriented objects extracted from facts, respectively representing a desired outcome, an explicit arrangement, and a closed-loop responsibility. Their status and target/start/due times are meaningful.
4. `episode`: a continuous-experience summary used only as an association boundary between facts. It must never be a direct recall object.

Guidance:
- `recall_object_types` lists object types eligible for direct retrieval. It must always include `fact`; add `entity_claim` only for explicit stable-entity-knowledge questions; add the relevant `goal`, `plan`, or `work_item` only for future goals, arrangements, responsibilities, deadlines, unfinished work, or their lifecycle. Never output `episode`.
- Prefer `fact` for what happened, dates, places, people, exact evidence, event order, and traceable details.
- For stable preferences, durable constraints, routines, relationships, or profiles, add `entity_claim` alongside the supporting facts.
- For tasks, commitments, future arrangements, goals, deadlines, or unfinished work, add the relevant prospective object alongside the supporting facts.
- Keep the plan broad when unsure. Missing evidence is worse than retrieving a few extra candidates.
- Extract 2-8 short retrieval keywords, prioritizing concrete people, organizations, products, projects, topics, actions, outcomes, and constraints. Do not output full sentences, pleasantries, generic words, or ordinary time expressions.
- Extract useful semantic entities with names and types. Entities may be people, organizations, locations, products, projects, technologies, or concrete concepts; ordinary time expressions such as today, yesterday, or last week are not entities.
- `temporal_mode` selects which fact timestamp should be used for a time range: `event_time` means the real-world event time described by the fact, `dialogue_time` means when the conversation/transcript occurred, `both` means either timestamp may match, and `none` means no hard time filter. Prefer `event_time` for queries asking what happened, was done, bought, or visited; prefer `dialogue_time` for queries asking what was discussed, mentioned, or asked; use `none` when the temporal intent is unclear.
- Parse `temporal_bounds` from the original query using the reference time. Use `YYYY-MM-DD HH:MM:SS` or `null` for `start` / `end`, and provide at least one bound. `end` is exclusive. Output `null` when no time constraint applies.
- Set `is_prospective_time_slot_query` to true only when the user asks for the set or status of their own current/future arrangements, plans, to-dos, or unfinished responsibilities in a time window. It is false for external information, advice, a single-fact confirmation, a prediction, or an immediate command such as creating a reminder.

Return JSON only:
{
  "recall_object_types": ["fact"],
  "needs_broad_evidence": false,
  "query_rewrite": "retrieval-focused rewrite over the unified memory projection",
  "keywords": ["keyword1", "keyword2"],
  "entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|OTHER"}],
  "temporal_bounds": {"start": "YYYY-MM-DD HH:MM:SS|null", "end": "YYYY-MM-DD HH:MM:SS|null"},
  "temporal_mode": "event_time|dialogue_time|both|none",
  "is_prospective_time_slot_query": false
}

Original user query:
{query}

Reference time for resolving relative time expressions:
{reference_time}
"""
