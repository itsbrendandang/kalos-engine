"""The grouped cross-validation's folds, precomputed in the worker pool while the
analysis fits its proposal model. The CV itself still runs after the proposal
fit, in the analysis thread, as before: it takes each precomputed fold that
started from the RNG state it has reached and fits the rest, so its numbers are
those of a CV run without the pool.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


class BackgroundCV:
    def __init__(self, call: Callable[[], dict]):
        from kalos.core.evaluation import cv_uses_pool, precomputing_folds

        self._call = call
        self._cache: dict = {}
        self._job: Future | None = None
        if cv_uses_pool():
            import torch

            start, threads = torch.get_rng_state(), torch.get_num_threads()

            def precompute() -> None:
                with precomputing_folds(self._cache, start, threads):
                    call()

            runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kalos-cv")
            self._job = runner.submit(precompute)
            runner.shutdown(wait=False)

    def result(self) -> dict:
        from kalos.core.evaluation import using_precomputed_folds

        self.settle()  # a fold that failed from the precompute state is refit below
        cache, self._cache = self._cache, {}
        with using_precomputed_folds(cache):
            return self._call()

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
