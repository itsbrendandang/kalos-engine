"""The default surrogate is a GP. Changing it requires beating the GP on the
published media datasets first - see "Surrogate guardrail" in BENCHMARK.md."""
import numpy as np
from botorch.models import SingleTaskGP

from kalos.core.surrogate import Surrogate


def test_default_surrogate_is_a_single_task_gp():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 2))
    s = Surrogate().fit(X, X.sum(1), bounds=np.array([[0, 0], [1, 1]], float))
    assert type(s.model) is SingleTaskGP
