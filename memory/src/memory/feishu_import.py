"""Single-file imports through the running owner's sole PWM writer.

Private durable receipts prevent duplicate/restarted imports from silently
writing the same historical meeting twice. A failed/interrupted import may
have partial writes and is never retried automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .feishu_transcript import OWNER_SPEAKER, parse_meeting_text

MAX_BYTES = 512 * 1024
JOB_ID = re.compile(r'^[a-f0-9]{64}$')
TERMINAL = {'completed', 'partial', 'failed', 'interrupted'}


def document(params):
    filename, content = params.get('filename'), params.get('content')
    if not isinstance(filename, str) or not filename.lower().endswith('.md') or len(filename) > 240:
        raise ValueError('invalid filename')
    if any(char in filename for char in ('/', '\\', '\0')):
        raise ValueError('invalid filename')
    if not isinstance(content, str) or not content.strip() or len(content.encode('utf-8')) > MAX_BYTES or '\0' in content:
        raise ValueError('invalid content')
    return filename, content.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')


def summary(meeting):
    return {key: meeting[key] for key in ('meeting_title', 'session_start', 'last_utterance_start',
                                          'source_segment_count', 'speakers', 'timestamp_policy')}


class FeishuMeetingImports:
    def __init__(self, *, owner_directory, get_pwm, operation_lock, ambient_active, logger):
        self.owner_directory, self.get_pwm = owner_directory, get_pwm
        self.operation_lock, self.ambient_active, self.logger = operation_lock, ambient_active, logger
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='feishu-import')
        self.active_owner = None
        self.active_id = None
        self.closed = False

    def busy(self, owner):
        with self.lock:
            return self.active_owner == owner

    @staticmethod
    def preview(params):
        try:
            filename, content = document(params)
            return {'ok': True, **summary(parse_meeting_text(content, filename=filename))}
        except (ValueError, TypeError, OverflowError):
            return {'ok': False, 'code': 'invalid_feishu_document'}

    def _directory(self, owner, job_id):
        if not isinstance(owner, str) or not owner.strip() or not JOB_ID.fullmatch(str(job_id)):
            raise ValueError('invalid import identity')
        return self.owner_directory(owner) / 'meeting-imports' / job_id

    @staticmethod
    def _save(directory, job):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = directory / 'job.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(job, stream, ensure_ascii=False)
        temporary.replace(directory / 'job.json')

    def status(self, owner, job_id):
        with self.lock:
            directory = self._directory(owner, job_id)
            path = directory / 'job.json'
            if not path.exists():
                return {'ok': False, 'code': 'feishu_import_not_found'}
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['status'] not in TERMINAL and (self.active_owner != owner or self.active_id != job_id):
                job.update(status='interrupted', phase='interrupted', errorCode='import_interrupted')
                self._save(directory, job)
            return {'ok': True, **job}

    def start(self, owner, params):
        try:
            filename, content = document(params)
            name = params.get('ownerName')
            if not isinstance(name, str) or not name.strip() or len(name) > 120:
                raise ValueError('select the owner explicitly')
            meeting = parse_meeting_text(content, filename=filename, owner_name=name)
        except (ValueError, TypeError, OverflowError):
            return {'ok': False, 'code': 'invalid_feishu_document'}
        job_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
        with self.lock:
            directory = self._directory(owner, job_id)
            if (directory / 'job.json').exists():
                return {**self.status(owner, job_id), 'duplicate': True}
            if self.closed or self.active_owner is not None or self.ambient_active(owner):
                return {'ok': False, 'code': 'feishu_import_busy'}
            job = {'id': job_id, 'filename': filename, 'ownerName': name,
                   'status': 'queued', 'phase': 'queued', 'createdAt': datetime.now().astimezone().isoformat(),
                   **summary(meeting)}
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(directory / 'source.md', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                stream.write(content)
            for segment in meeting['segments']:
                segment['metadata'].update(source_file=str(directory / 'source.md'), import_id=job_id)
            self._save(directory, job)
            self.active_owner = owner
            self.active_id = job_id
            self.executor.submit(self._run, owner, directory, meeting, job)
            return {'ok': True, **job}

    def _update(self, directory, job, **values):
        with self.lock:
            job.update(values, updatedAt=datetime.now().astimezone().isoformat())
            self._save(directory, job)

    @staticmethod
    def _accepted(report, *, root=True):
        rejected = {'memory_disabled', 'queue_rejected', 'worker_queue_full', 'worker_shutdown',
                    'task_handler_not_registered'}
        if isinstance(report, dict):
            if (root and report.get('accepted') is False) or report.get('reason') in rejected:
                raise RuntimeError('PWM rejected meeting input or finalization')
            for value in report.values():
                if isinstance(value, dict):
                    FeishuMeetingImports._accepted(value, root=False)

    @staticmethod
    def _register_selected_owner(pwm, name):
        # The operator explicitly selected this identity; no fabricated fact
        # is introduced. Schema permits NULL evidence for direct configuration.
        db = pwm.database
        canonical = pwm.memory_manager.world_owner_entity_name
        if name == canonical:
            return
        with pwm.task_scheduler.execution_lock, db.transaction():
            entity_id = db.upsert_entity_nodes([{'name': canonical, 'type': 'PERSON'}])[canonical]
            now = datetime.now().astimezone().isoformat()
            db._conn.execute('''INSERT INTO memory_owner_entity_aliases (
                owner_entity_id, alias_name, alias_key, status, confidence, evidence_fact_id,
                evidence_type, evidence_text, source_speaker, created_at, updated_at
            ) VALUES (?, ?, ?, 'confirmed', 1.0, NULL, 'meeting_import_selection', ?, ?, ?, ?)
            ON CONFLICT(owner_entity_id, alias_key) DO UPDATE SET status='confirmed', confidence=1.0,
                evidence_fact_id=NULL, evidence_type=excluded.evidence_type,
                evidence_text=excluded.evidence_text, source_speaker=excluded.source_speaker,
                updated_at=excluded.updated_at''',
                (entity_id, name, db.normalize_owner_alias_key(name),
                 'User explicitly selected their speaker identity for a meeting import.', OWNER_SPEAKER, now, now))

    def _run(self, owner, directory, meeting, job):
        started = False
        pwm = None
        try:
            self._update(directory, job, status='running', phase='preparing')
            with self.operation_lock:
                pwm = self.get_pwm(owner)
                self._accepted(pwm.finalize_memory_session(reason='before_feishu_import'))
            if not pwm.wait_for_memory_tasks():
                raise RuntimeError('prior memory tasks did not drain')
            reporter = pwm.task_scheduler.operation_reporter
            before = reporter.snapshot()
            self._register_selected_owner(pwm, meeting['owner_name'])
            tags = ['feishu_meeting', 'meeting_import', f"meeting:{meeting['meeting_title']}", f"import:{job['id']}"]
            self._update(directory, job, phase='extracting')
            started = True
            self._accepted(pwm.accept_memory_input(transcript_segments=meeting['segments'], tags=tags,
                                                   ambient_recording_enabled=True))
            self._accepted(pwm.finalize_memory_session(reason='feishu_meeting_complete', tags=tags))
            if not pwm.wait_for_memory_tasks():
                raise RuntimeError('meeting extraction tasks did not drain')
            self._update(directory, job, phase='updating_states')
            self._accepted(pwm.drain_pending_user_state_claims())
            if not pwm.wait_for_memory_tasks():
                raise RuntimeError('meeting state tasks did not drain')
            after = reporter.snapshot()
            previous_ids = {item['task_id'] for item in before['recent_tasks']}
            failed = sum(max(0, values.get('failed', 0) - before['counts'].get(kind, {}).get('failed', 0))
                         for kind, values in after['counts'].items())
            partial = any(item.get('status') == 'partial' and item['task_id'] not in previous_ids
                          for item in after['recent_tasks'])
            self._update(directory, job, status='partial' if failed or partial else 'completed',
                         phase='completed', failedTasks=failed)
            self.logger.info('feishu meeting import finished id=%s status=%s segments=%s',
                             job['id'][:12], job['status'], meeting['source_segment_count'])
        except Exception:
            self.logger.exception('feishu meeting import failed id=%s', job['id'][:12])
            if pwm is not None and started:
                try:
                    pwm.finalize_memory_session(reason='feishu_import_failed')
                    pwm.wait_for_memory_tasks()
                except Exception:
                    self.logger.exception('feishu import tail cleanup failed id=%s', job['id'][:12])
            self._update(directory, job, status='partial' if started else 'failed', phase='failed', errorCode='feishu_extraction_failed')
        finally:
            with self.lock:
                self.active_owner = None
                self.active_id = None

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
