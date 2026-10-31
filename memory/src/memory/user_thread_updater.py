"""Activity-centered matter tracking: resolve membership, then update progress.

This is a UserMemory helper, not a UserModel/state resolver. The shared FIFO
serializes its reads and writes with other memory tasks.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence

from prompts_en import USER_THREAD_MEMBERSHIP_PROMPT_EN, USER_THREAD_UPDATE_PROMPT_EN
from prompts_zh import USER_THREAD_MEMBERSHIP_PROMPT_ZH, USER_THREAD_UPDATE_PROMPT_ZH

from .memory_database import UserMemoryDB, _lexical_index_text


class UserThreadUpdater:
    """Use two separate LLM judgments; commit only fully validated batches."""

    def __init__(
        self, *, memory_db: UserMemoryDB, llm_client: Any,
        resolve_language: Callable[[str], str], log: Callable[..., None],
        batch_size: int = 8, candidate_limit: int = 12, event_context_limit: int = 16,
    ) -> None:
        self._db = memory_db
        self._llm_client = llm_client
        self._resolve_language = resolve_language
        self._log = log
        self._batch_size = max(1, int(batch_size))
        self._candidate_limit = max(1, int(candidate_limit))
        self._event_context_limit = max(1, int(event_context_limit))

    @staticmethod
    def _ids(values: Any) -> List[int]:
        if not isinstance(values, list):
            raise ValueError('Expected an ID list')
        if any(isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0
               for value in values):
            raise ValueError('Expected positive integer IDs')
        return list(dict.fromkeys(int(value) for value in values))

    @staticmethod
    def _text(row: Dict[str, Any], key: str) -> str:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Missing nonempty {key}')
        return value.strip()

    @staticmethod
    def _activity_view(activity: Dict[str, Any]) -> Dict[str, Any]:
        return {key: activity.get(key) for key in (
            'id', 'activity_text', 'user_role', 'confidence', 'started_at', 'ended_at',
        )}

    def _build_activity_cases(self, activity_ids: Sequence[int]) -> List[Dict[str, Any]]:
        activities = self._db.get_user_activity_items_by_ids(activity_ids)
        if {row['id'] for row in activities} != set(activity_ids):
            raise ValueError('Unknown user activity IDs')
        cases = []
        for activity in activities:
            facts = self._db.get_memory_facts_by_ids(activity['evidence_fact_ids'])
            fact_views = [{key: fact.get(key) for key in (
                'id', 'summary', 'fact_topic', 'entities', 'keywords',
                'dialogue_time_key', 'event_time_key',
            )} for fact in facts]
            search_text = '\n'.join([
                activity['activity_text'],
                *[str(fact.get('fact_topic') or '') for fact in facts],
                *[str(fact.get('summary') or '') for fact in facts],
            ])
            structured_terms = [
                str(value).strip() for fact in facts
                for field in ('entities', 'keywords') for value in fact.get(field) or []
            ]
            terms = [term for term in [*structured_terms, *_lexical_index_text(search_text).split()]
                     if len(term) >= 2 and term.lower() not in {
                         '用户', '用户本人', '助手', '讨论', '交谈', 'the', 'user', 'assistant',
                     }]
            # Specific phrases precede generic short tokens; retrieval is deliberately
            # broad, with the LLM responsible for identity and user involvement.
            terms = sorted(set(terms), key=lambda term: (-len(term), term))[:32]
            cases.append({
                'activity': self._activity_view(activity), 'linked_facts': fact_views,
                'candidate_threads': self._db.search_user_threads(
                    terms, activity_id=activity['id'], limit=self._candidate_limit,
                ),
            })
        return cases

    def _call_llm(self, stage: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = json.dumps(cases, ensure_ascii=False, default=str)
        language = self._resolve_language(text)
        templates = {
            'membership': (USER_THREAD_MEMBERSHIP_PROMPT_ZH, USER_THREAD_MEMBERSHIP_PROMPT_EN),
            'update': (USER_THREAD_UPDATE_PROMPT_ZH, USER_THREAD_UPDATE_PROMPT_EN),
        }
        template = templates[stage][language == 'en']
        field = 'activity_cases' if stage == 'membership' else 'thread_cases'
        self._log('memory_user_thread_update', f'{stage}_input', {
            'prompt_language': language, field: cases,
        })
        response = self._llm_client.call_llm(template.format(**{field: text}))
        result = self._llm_client.parse_json_object_from_llm_text(response or '')
        self._log('memory_user_thread_update', f'{stage}_result', {'result': result})
        if not isinstance(result, dict):
            raise ValueError(f'Invalid user thread {stage} response')
        return result

    def _build_thread_cases(
        self, cases: List[Dict[str, Any]], result: Dict[str, Any],
        reference_timestamp: Any,
    ) -> List[Dict[str, Any]]:
        definitions = {}
        for row in result.get('new_threads', []):
            ref = self._text(row, 'thread_ref')
            if not ref.startswith('new:') or not ref[4:] or ref in definitions:
                raise ValueError('Invalid or duplicate new thread reference')
            definitions[ref] = {
                'title': self._text(row, 'title'),
                'identity_description': self._text(row, 'identity_description'),
            }
        cases_by_id = {row['activity']['id']: row for row in cases}
        grouped: Dict[str, Dict[str, Any]] = {}
        seen = set()
        for assignment in result.get('activity_assignments', []):
            activity_id = self._ids([assignment['activity_id']])[0]
            if activity_id not in cases_by_id or activity_id in seen:
                raise ValueError('Unknown or duplicate activity assignment')
            seen.add(activity_id)
            case = cases_by_id[activity_id]
            existing = {f"existing:{row['id']}": row for row in case['candidate_threads']}
            facts = {row['id']: row for row in case['linked_facts']}
            refs = assignment.get('thread_refs')
            if not isinstance(refs, list):
                raise ValueError('Missing thread_refs list')
            if not refs:
                self._text(assignment, 'skip_reason')
            seen_refs = set()
            for membership in refs:
                ref = self._text(membership, 'thread_ref')
                self._text(membership, 'reason')
                if ref in seen_refs or (ref not in definitions and ref not in existing):
                    raise ValueError('Unknown or duplicate thread membership')
                seen_refs.add(ref)
                fact_ids = self._ids(membership.get('fact_ids'))
                if not set(fact_ids) <= facts.keys():
                    raise ValueError('Thread membership references unrelated facts')
                target = grouped.setdefault(ref, {
                    'thread_ref': ref, 'existing_thread': existing.get(ref),
                    'proposed_identity': definitions.get(ref), 'new_activity_cases': [],
                    'recent_events': self._db.get_user_thread_events(
                        existing[ref]['id'], limit=self._event_context_limit,
                    ) if ref in existing else [],
                    'reference_time': reference_timestamp,
                })
                target['new_activity_cases'].append({
                    'activity': case['activity'],
                    'assigned_facts': [facts[value] for value in fact_ids],
                })
        if seen != cases_by_id.keys():
            raise ValueError('LLM omitted activity assignments')
        if not definitions.keys() <= grouped.keys():
            raise ValueError('Unreferenced new thread definition')
        return list(grouped.values())

    def _validate_updates(
        self, cases: List[Dict[str, Any]], result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        cases_by_ref = {case['thread_ref']: case for case in cases}
        seen = set()
        updates = result.get('thread_updates')
        if not isinstance(updates, list):
            raise ValueError('Missing thread_updates list')
        for update in updates:
            ref = self._text(update, 'thread_ref')
            if ref not in cases_by_ref or ref in seen:
                raise ValueError('Unknown or duplicate thread update')
            seen.add(ref)
            for field in ('title', 'identity_description', 'current_summary'):
                update[field] = self._text(update, field)
            if update.get('status') not in {'active', 'closed'}:
                raise ValueError('Invalid thread status')
            case = cases_by_ref[ref]
            evidence_by_activity = {
                row['activity']['id']: {fact['id'] for fact in row['assigned_facts']}
                for row in case['new_activity_cases']
            }
            events = update.get('events')
            if not isinstance(events, list):
                raise ValueError('Missing events list')
            event_types = set()
            for event in events:
                if event.get('event_type') not in {
                    'progress', 'change', 'blocked', 'resolved', 'closed', 'reopened',
                }:
                    raise ValueError('Invalid thread event type')
                event_types.add(event['event_type'])
                event['summary'] = self._text(event, 'summary')
                activities = self._ids(event.get('evidence_activity_ids'))
                facts = self._ids(event.get('evidence_fact_ids'))
                if not activities or not set(activities) <= evidence_by_activity.keys():
                    raise ValueError('Thread event requires scoped new activity evidence')
                allowed_facts = set().union(*(evidence_by_activity[value] for value in activities))
                if not set(facts) <= allowed_facts:
                    raise ValueError('Thread event references unrelated fact evidence')
                event['evidence_activity_ids'], event['evidence_fact_ids'] = activities, facts
                occurred_at = event.get('occurred_at')
                if occurred_at is not None:
                    if not isinstance(occurred_at, str):
                        raise ValueError('Invalid thread occurrence time')
                    occurred_at = occurred_at.strip()
                    if not occurred_at:
                        occurred_at = None
                    elif re.fullmatch(r'\d{4}(?:-\d{2})?', occurred_at):
                        # Validate coarse calendar values without persisting an
                        # invented day/time (e.g. keep "2023-05", not May 1).
                        datetime.strptime(
                            occurred_at, '%Y' if len(occurred_at) == 4 else '%Y-%m',
                        )
                    else:
                        datetime.fromisoformat(occurred_at.replace('Z', '+00:00'))
                    event['occurred_at'] = occurred_at
            old_status = (case['existing_thread'] or {}).get('status', 'active')
            if update['status'] != old_status:
                required = 'closed' if update['status'] == 'closed' else 'reopened'
                if required not in event_types:
                    raise ValueError('Thread lifecycle change requires a grounded event')
            if ('closed' in event_types and update['status'] != 'closed') or (
                'reopened' in event_types and update['status'] != 'active'
            ):
                raise ValueError('Thread status contradicts lifecycle event')
            if 'reopened' in event_types and old_status != 'closed':
                raise ValueError('Only a previously closed thread can reopen')
        if seen != cases_by_ref.keys():
            raise ValueError('LLM omitted thread updates')
        return updates

    def _save_updates(
        self, cases: List[Dict[str, Any]], updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report = {
            'new_thread_ids': [], 'updated_thread_ids': [],
            'thread_event_ids': [], 'thread_changes': [],
        }
        by_ref = {case['thread_ref']: case for case in cases}
        with self._db.transaction():
            for update in updates:
                case = by_ref[update['thread_ref']]
                old = case['existing_thread']
                activities = [row['activity']['id'] for row in case['new_activity_cases']]
                facts = sorted({fact['id'] for row in case['new_activity_cases']
                                for fact in row['assigned_facts']})
                creation_key = hashlib.sha256(json.dumps(
                    [sorted(activities), update['thread_ref']], ensure_ascii=False,
                ).encode('utf-8')).hexdigest()
                thread_id = self._db.save_user_thread(
                    thread_id=old['id'] if old else None, creation_key=creation_key,
                    **{field: update[field] for field in (
                        'title', 'identity_description', 'current_summary', 'status',
                    )},
                )
                report['updated_thread_ids' if old else 'new_thread_ids'].append(thread_id)
                self._db.link_user_thread_evidence(
                    thread_id, activity_ids=activities, fact_ids=facts,
                )
                known_at = str(case['reference_time'])
                event_ids = []
                for event in update['events']:
                    event_id = self._db.insert_user_thread_event(
                        thread_id, known_at=known_at,
                        **{field: event.get(field) for field in (
                            'event_type', 'summary', 'evidence_activity_ids',
                            'evidence_fact_ids', 'occurred_at',
                        )},
                    )
                    if event_id is not None:
                        report['thread_event_ids'].append(event_id)
                        event_ids.append(event_id)
                identity_changed = old is not None and any(
                    old[field] != update[field]
                    for field in ('identity_description', 'status')
                )
                if old is None or event_ids or identity_changed:
                    # Freeze this committed update, not a future version read by
                    # the state task after other thread tasks in the FIFO.
                    fields = ('title', 'identity_description', 'current_summary', 'status')
                    report['thread_changes'].append({
                        'thread': {'id': thread_id, **{key: update[key] for key in fields}},
                        'previous_thread': (
                            {'id': thread_id, **{key: old[key] for key in fields}}
                            if old is not None else None
                        ),
                        'thread_event_ids': event_ids,
                        'evidence_activity_ids': sorted(set(activities)),
                        'evidence_fact_ids': facts,
                        'reference_time': known_at,
                    })
        return report

    def process_update_task(
        self, *, activity_ids: List[int], reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        ids = self._ids(activity_ids)
        report: Dict[str, Any] = {
            'status': 'ok' if ids else 'empty', 'processed_activity_ids': [],
            'new_thread_ids': [], 'updated_thread_ids': [], 'thread_event_ids': [],
            'thread_changes': [],
        }
        for offset in range(0, len(ids), self._batch_size):
            batch_ids = ids[offset:offset + self._batch_size]
            try:
                cases = self._build_activity_cases(batch_ids)
                # Use the source clock in replay tests; unknown real occurrence
                # times remain null rather than being fabricated from this clock.
                source_times = [row['activity'].get('ended_at') for row in cases]
                clock = reference_timestamp or max(
                    (value for value in source_times if value), default=None,
                    key=lambda value: datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(),
                ) or datetime.now().astimezone().isoformat()
                membership = self._call_llm('membership', cases)
                thread_cases = self._build_thread_cases(cases, membership, clock)
                if thread_cases:
                    result = self._call_llm('update', thread_cases)
                    updates = self._validate_updates(thread_cases, result)
                    saved = self._save_updates(thread_cases, updates)
                    for key, values in saved.items():
                        report[key].extend(values)
                report['processed_activity_ids'].extend(batch_ids)
            except Exception as exc:
                report.update(
                    status='failed', reason=str(exc), error_type=type(exc).__name__,
                    failed_activity_ids=batch_ids,
                )
                self._log('memory_user_thread_update', 'failed', report)
                return report
        self._log('memory_user_thread_update', 'finish', report)
        return report
