from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Literal, Optional, Tuple


JobStatus = Literal["running", "done", "error", "canceled"]
JobKind = Literal["tail", "bash"]


@dataclass
class OutputEvent:
    seq: int
    stream: str
    text: str


@dataclass
class AsyncJob:
    id: str
    kind: JobKind
    created_at: float
    status: JobStatus = "running"
    last_update_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    # Output storage (bounded).
    _events: Deque[OutputEvent] = field(default_factory=deque)
    _event_chars: int = 0
    _next_seq: int = 1

    # Kind-specific state.
    tail_path: Optional[Path] = None
    tail_offset: int = 0
    tail_encoding: str = "utf-8"

    proc: Optional[subprocess.Popen] = None
    proc_cwd: Optional[Path] = None


class AsyncJobManager:
    def __init__(self, *, max_events: int = 500, max_event_chars: int = 50_000) -> None:
        self._lock = threading.Lock()
        self._counter = 1
        self._jobs: Dict[str, AsyncJob] = {}
        self._max_events = int(max_events)
        self._max_event_chars = int(max_event_chars)

    def _new_id(self) -> str:
        with self._lock:
            job_id = f"job_{self._counter}"
            self._counter += 1
            return job_id

    def list_jobs(self) -> List[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [self._job_summary(j) for j in jobs]

    def get_job(self, job_id: str) -> Optional[AsyncJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start_tail(self, path: Path, *, start_at: Literal["start", "end"] = "end", encoding: str = "utf-8") -> str:
        job_id = self._new_id()
        created_at = time.time()
        job = AsyncJob(id=job_id, kind="tail", created_at=created_at, tail_path=path, tail_encoding=encoding)

        try:
            stat_size = path.stat().st_size
        except OSError:
            stat_size = 0
        job.tail_offset = 0 if start_at == "start" else int(stat_size)

        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def start_bash(self, command: str, *, cwd: Path) -> str:
        job_id = self._new_id()
        created_at = time.time()
        job = AsyncJob(id=job_id, kind="bash", created_at=created_at, proc_cwd=cwd)

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            job.proc = proc
        except (OSError, ValueError) as exc:
            job.status = "error"
            job.error = f"async_bash exception: {exc}"
            with self._lock:
                self._jobs[job_id] = job
            return job_id

        with self._lock:
            self._jobs[job_id] = job

        def reader(stream_name: str, pipe) -> None:
            try:
                if pipe is None:
                    return
                for line in pipe:
                    self._append_event(job_id, stream_name, line.rstrip("\n"))
            except OSError as exc:
                self._set_error(job_id, f"{stream_name} reader error: {exc}")

        threading.Thread(target=reader, args=("stdout", proc.stdout), daemon=True).start()
        threading.Thread(target=reader, args=("stderr", proc.stderr), daemon=True).start()
        threading.Thread(target=self._watch_process, args=(job_id,), daemon=True).start()
        return job_id

    def stop(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            if job.status != "running":
                return True
            job.status = "canceled"
            job.last_update_at = time.time()
        if job.kind == "bash" and job.proc is not None:
            try:
                job.proc.terminate()
            except OSError:
                pass
        return True

    def poll(self, job_id: str, *, since_seq: int = 0) -> Optional[dict]:
        job = self.get_job(job_id)
        if job is None:
            return None

        if job.kind == "tail":
            self._poll_tail(job)

        events = self._events_since(job, since_seq)
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "error": job.error,
            "since_seq": since_seq,
            "next_seq": job._next_seq,
            "events": [{"seq": e.seq, "stream": e.stream, "text": e.text} for e in events],
        }

    def _job_summary(self, job: AsyncJob) -> dict:
        detail: dict = {"id": job.id, "kind": job.kind, "status": job.status, "error": job.error}
        if job.kind == "tail" and job.tail_path is not None:
            detail["path"] = str(job.tail_path)
        if job.kind == "bash" and job.proc is not None:
            detail["pid"] = getattr(job.proc, "pid", None)
        return detail

    def _events_since(self, job: AsyncJob, since_seq: int) -> List[OutputEvent]:
        with self._lock:
            return [e for e in job._events if e.seq > since_seq]

    def _append_event(self, job_id: str, stream: str, text: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "running":
                return
            seq = job._next_seq
            job._next_seq += 1
            event = OutputEvent(seq=seq, stream=stream, text=text)
            job._events.append(event)
            job._event_chars += len(text)
            job.last_update_at = time.time()
            while len(job._events) > self._max_events or job._event_chars > self._max_event_chars:
                old = job._events.popleft()
                job._event_chars -= len(old.text)

    def _set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == "running":
                job.status = "error"
            job.error = error
            job.last_update_at = time.time()

    def _watch_process(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None or job.proc is None:
            return
        proc = job.proc
        try:
            code = proc.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            self._set_error(job_id, f"process wait error: {exc}")
            return
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            if current.status == "canceled":
                return
            if code == 0:
                current.status = "done"
            else:
                current.status = "error"
                current.error = f"exit {code}"
            current.last_update_at = time.time()
        self._append_event(job_id, "meta", f"process exited with code {code}")

    def _poll_tail(self, job: AsyncJob) -> None:
        if job.tail_path is None:
            return
        if job.status != "running":
            return
        path = job.tail_path
        try:
            with path.open("r", encoding=job.tail_encoding, errors="replace") as f:
                f.seek(job.tail_offset)
                data = f.read()
                job.tail_offset = f.tell()
        except (OSError, LookupError, UnicodeError, ValueError) as exc:
            self._set_error(job.id, f"tail read error: {exc}")
            return
        if not data:
            return
        for line in data.splitlines():
            self._append_event(job.id, "tail", line)


_DEFAULT_MANAGER: Optional[AsyncJobManager] = None


def get_async_job_manager() -> AsyncJobManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = AsyncJobManager()
    return _DEFAULT_MANAGER
