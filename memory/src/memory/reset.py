"""Owner-scoped reset, with a durable fence outside the databases being erased."""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class MemoryResets:
    def __init__(self, *, root, owner_key, operation_lock, owners, get_owner, imports, logger):
        self.root, self.owner_key = root, owner_key
        self.operation_lock, self.owners, self.get_owner = operation_lock, owners, get_owner
        self.imports, self.logger = imports, logger
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='memory-reset')
        self.active = set()
        self.closed = False

    def _path(self, owner):
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError('ownerId must be non-empty')
        return self.root / 'memory-resets' / (self.owner_key(owner) + '.json')

    def _read(self, owner):
        path = self._path(owner)
        if not path.exists():
            return None
        job = json.loads(path.read_text(encoding='utf-8'))
        if job['status'] in {'queued', 'running'} and owner not in self.active:
            job.update(status='interrupted', phase='interrupted')
            self._save(owner, job)
        return job

    def _save(self, owner, job):
        path = self._path(owner)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix('.tmp')
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(job, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name != 'nt':
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def status(self, owner):
        with self.lock:
            job = self._read(owner)
            return {'ok': True, 'generation': job['generation'] if job else '0', 'job': job}

    def blocked(self, owner):
        with self.lock:
            job = self._read(owner)
            return bool(job and job['status'] != 'completed')

    def accepts(self, owner, generation):
        with self.lock:
            state = self.status(owner)
            return not self.blocked(owner) and str(generation or '0') == state['generation']

    def start(self, owner):
        # Caller holds the operation lock, so input/import/recording reservation
        # cannot slip between the busy check and the persisted fence.
        with self.lock:
            previous = self._read(owner)
            if owner in self.active:
                return {'ok': True, 'duplicate': True, 'generation': previous['generation'], 'job': previous}
            holder = self.owners.get(self.owner_key(owner))
            if self.closed or self.imports.busy(owner) or (holder and holder.active_ambient_recording_ids):
                return {'ok': False, 'code': 'memory_reset_busy'}
            now = datetime.now().astimezone().isoformat()
            job = {'id': uuid.uuid4().hex, 'generation': uuid.uuid4().hex,
                   'status': 'queued', 'phase': 'waiting', 'createdAt': now}
            self.active.add(owner)
            try:
                self._save(owner, job)
                self.executor.submit(self._run, owner, job)
            except Exception:
                self.active.discard(owner)
                raise
            return {'ok': True, 'generation': job['generation'], 'job': dict(job)}

    def _update(self, owner, job, **values):
        with self.lock:
            job.update(values, updatedAt=datetime.now().astimezone().isoformat())
            self._save(owner, job)

    def _run(self, owner, job):
        try:
            self._update(owner, job, status='running', phase='waiting')
            key = self.owner_key(owner)
            with self.operation_lock:
                holder = self.owners.get(key)
            # Do not hold the IPC lock or reset lock while waiting on model calls.
            if holder:
                holder.pwm.close(timeout=None)
            with self.operation_lock:
                self.owners.pop(key, None)
                self._update(owner, job, phase='clearing')
                directory = self.root / 'owners' / key
                # Only PWM-owned data; never remove settings, voice samples,
                # source meeting files or ambient WAVs outside this directory.
                for name in ('memory.db', 'user_model.db', 'task_scheduler.db'):
                    for suffix in ('', '-wal', '-shm', '-journal'):
                        (directory / (name + suffix)).unlink(missing_ok=True)
                for name in ('observed-message-ids.json', 'observed-message-ids.tmp'):
                    (directory / name).unlink(missing_ok=True)
                imports = directory / 'meeting-imports'
                if imports.is_symlink():
                    imports.unlink()
                elif imports.exists():
                    shutil.rmtree(imports)
                self._update(owner, job, phase='initializing')
                self.get_owner(owner)
                self._update(owner, job, status='completed', phase='completed')
            self.logger.info('memory reset completed owner=%s id=%s', key[:12], job['id'])
        except Exception:
            self.logger.exception('memory reset failed id=%s', job['id'])
            self._update(owner, job, status='failed', phase='failed')
        finally:
            with self.lock:
                self.active.discard(owner)

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
