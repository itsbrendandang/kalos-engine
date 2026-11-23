"""The grouped cross-validation, started before the proposal surrogate is fit so
the two overlap: the folds fit in the worker pool while the analysis fits its
proposal model. Without a pool the CV runs when its result is asked for, after
the proposal fit, as before.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


class BackgroundCV:
    def __init__(self, call: Callable[[], dict]):
        self._call = call
        self._job: Future | None = None
        from kalos.core.evaluation import cv_uses_pool  # torch: not at import time

        if cv_uses_pool():
            runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kalos-cv")
            self._job = runner.submit(call)
            runner.shutdown(wait=False)

    def result(self) -> dict:
        return self._job.result() if self._job is not None else self._call()

    def settle(self) -> None:
        if self._job is not None:
            self._job.exception()


@contextlib.contextmanager
def cv_alongside(call: Callable[[], dict]) -> Iterator[BackgroundCV]:
    """Start the CV and run the body (the proposal fit) alongside it. If the body
    fails, wait for the folds before re-raising, so the analysis slot is held
    until its CV work is done and no fold fits run outside the admission limit."""
    cv = BackgroundCV(call)
    try:
        yield cv
    except BaseException:
        cv.settle()
        raise
