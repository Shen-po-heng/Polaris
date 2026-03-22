"""Task queue — async document indexing with bounded concurrency.

Runs a background worker pool (max 2 concurrent jobs) so the Gradio UI
stays responsive while large PDFs are being indexed.

Usage:
    import asyncio
    from services.task_queue import IndexQueue

    queue = IndexQueue(rag_service)
    asyncio.run(queue.submit(["paper.pdf"]))  # non-blocking enqueue
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from core.exceptions import IndexingError
from utils.logger import get_logger

if TYPE_CHECKING:
    from services.rag_service import RAGService

logger = get_logger(__name__)

_MAX_WORKERS = 2


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class IndexJob:
    file_paths: list[str]
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    result: object = field(default=None, repr=False)


class IndexQueue:
    """Bounded async queue for document indexing jobs.

    Args:
        rag_service: The ``RAGService`` instance to delegate indexing to.
        max_workers: Maximum number of concurrent indexing jobs.
    """

    def __init__(
        self, rag_service: RAGService, max_workers: int = _MAX_WORKERS
    ) -> None:
        self._rag = rag_service
        self._queue: asyncio.Queue[IndexJob] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_workers)
        self._started = False

    # ── Public API ──────────────────────────────────────────────────────────

    async def submit(self, file_paths: list[str]) -> IndexJob:
        """Enqueue a list of file paths for indexing.

        Returns the job immediately; check ``job.status`` to poll progress.
        """
        job = IndexJob(file_paths=file_paths)
        await self._queue.put(job)
        logger.info("Job enqueued: %d file(s)", len(file_paths))

        if not self._started:
            asyncio.ensure_future(self._worker())

        return job

    # ── Private helpers ─────────────────────────────────────────────────────

    async def _worker(self) -> None:
        self._started = True
        while True:
            job = await self._queue.get()
            asyncio.ensure_future(self._process(job))
            self._queue.task_done()

    async def _process(self, job: IndexJob) -> None:
        async with self._semaphore:
            job.status = JobStatus.RUNNING
            logger.info("Starting indexing job: %s", job.file_paths)
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._rag.process_document, job.file_paths
                )
                job.result = result
                job.status = JobStatus.DONE
                logger.info("Indexing job completed: %s", job.file_paths)
            except (IndexingError, Exception) as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                logger.error("Indexing job failed: %s — %s", job.file_paths, exc)
