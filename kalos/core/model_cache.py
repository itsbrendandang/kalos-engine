"""Content-addressed disk cache for fitted BoTorch GP surrogates.

Torch-free at module import time (`test_core_torch_contract.py` enforces this
for every ``kalos.core`` module outside the allowlist): ``torch``, ``botorch``,
and ``kalos.core.surrogate`` are imported lazily inside ``get`` and ``put``.

Security model:
  - Only tensor ``state_dict`` weights are saved via ``torch.save``, and they
    are loaded strictly with ``torch.load(..., map_location="cpu", weights_only=True)``.
  - Metadata (bounds, cat_dims, fit_diagnostics, shape, noise metadata) lives in
    a JSON sidecar; ``SingleTaskGP`` / ``MixedSingleTaskGP`` is reconstructed
    from scratch matching ``Surrogate.fit`` before ``load_state_dict(..., strict=True)``.
  - Writes are atomic (temp file + ``os.replace``), bounded by LRU eviction, and
    fail-soft (corrupt/incompatible entries log a warning and return ``None``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("kalos.core.model_cache")

_CACHE_FORMAT_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert numpy/non-JSON scalars into strict-JSON-safe Python primitives."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _canonical_inputs(
    X_fit: Any,
    y_fit: Any,
    bounds: Any,
    cat_dims: Sequence[int] | None,
    noise: float | np.ndarray | Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray | None]:
    Xa = np.ascontiguousarray(np.asarray(X_fit, dtype=np.float64))
    ya = np.ascontiguousarray(np.asarray(y_fit, dtype=np.float64).reshape(-1))
    ba = np.ascontiguousarray(np.asarray(bounds, dtype=np.float64))
    cats = [int(i) for i in cat_dims] if cat_dims else []
    noise_arr: np.ndarray | None = None
    if noise is not None:
        noise_arr = np.ascontiguousarray(
            np.broadcast_to(np.asarray(noise, dtype=np.float64), (Xa.shape[0],)).astype(np.float64)
        )
    return Xa, ya, ba, cats, noise_arr


def _hash_array(h: Any, tag: bytes, arr: np.ndarray) -> None:
    h.update(tag)
    h.update(struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        h.update(struct.pack("<Q", dim))
    h.update(arr.tobytes())


class SurrogateCache:
    """Content-addressed, bounded-LRU disk cache for fitted ``Surrogate`` models."""

    def __init__(self, cache_dir: Path | str, max_entries: int = 64) -> None:
        if int(max_entries) < 0:
            raise ValueError("max_entries must be non-negative")
        self.cache_dir = Path(cache_dir)
        self.max_entries = int(max_entries)
        self._last_ns = 0

    def _next_ns(self) -> int:
        now = time.time_ns()
        if now <= self._last_ns:
            now = self._last_ns + 1
        self._last_ns = now
        return now

    def _touch(self, *paths: Path) -> None:
        ns = self._next_ns()
        for p in paths:
            try:
                os.utime(p, ns=(ns, ns))
            except OSError:
                pass

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.pt", self.cache_dir / f"{key}.json"

    def cache_key(
        self,
        X_fit: Any,
        y_fit: Any,
        bounds: Any,
        *,
        cat_dims: Sequence[int] | None = None,
        noise: float | np.ndarray | Sequence[float] | None = None,
        restarts: int = 1,
        restart_seed: int = 0,
        check_conditioning: bool = False,
        engine_version: str | None = None,
    ) -> str:
        """Compute the content-addressed SHA-256 cache key for a fit configuration."""
        if engine_version is None:
            from kalos import __version__

            engine_version = __version__
        Xa, ya, ba, cats, noise_arr = _canonical_inputs(X_fit, y_fit, bounds, cat_dims, noise)
        h = hashlib.sha256()
        h.update(b"kalos.surrogate_cache.v1\x00")
        _hash_array(h, b"X", Xa)
        _hash_array(h, b"y", ya)
        _hash_array(h, b"B", ba)
        if noise_arr is not None:
            _hash_array(h, b"N", noise_arr)
        else:
            h.update(b"N:none\x00")
        meta = json.dumps(
            {
                "cat_dims": cats,
                "restarts": int(restarts),
                "restart_seed": int(restart_seed),
                "check_conditioning": bool(check_conditioning),
                "engine_version": str(engine_version),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        h.update(struct.pack("<Q", len(meta)))
        h.update(meta)
        return h.hexdigest()

    def get(
        self,
        X_fit: Any,
        y_fit: Any,
        bounds: Any,
        *,
        cat_dims: Sequence[int] | None = None,
        noise: float | np.ndarray | Sequence[float] | None = None,
        restarts: int = 1,
        restart_seed: int = 0,
        check_conditioning: bool = False,
        engine_version: str | None = None,
    ) -> Any | None:
        """Load and reconstruct a cached ``Surrogate``, or return ``None`` on miss/error."""
        if self.max_entries <= 0 or not self.cache_dir.exists():
            return None
        try:
            Xa, ya, ba, cats, noise_arr = _canonical_inputs(X_fit, y_fit, bounds, cat_dims, noise)
            key = self.cache_key(
                Xa, ya, ba, cat_dims=cats, noise=noise_arr, restarts=restarts,
                restart_seed=restart_seed, check_conditioning=check_conditioning,
                engine_version=engine_version,
            )
        except Exception as exc:
            log.warning("SurrogateCache key computation failed: %s", exc)
            return None

        weights_path, meta_path = self._paths(key)
        if not meta_path.exists() or not weights_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("cache metadata is not a JSON object")
            if meta.get("version") != _CACHE_FORMAT_VERSION or meta.get("key") != key:
                raise ValueError("cache metadata version or key mismatch")
            if meta.get("shape") != [int(Xa.shape[0]), int(Xa.shape[1])]:
                raise ValueError("cache shape mismatch")
            if meta.get("cat_dims") != cats:
                raise ValueError("cache cat_dims mismatch")
            has_noise = noise_arr is not None
            if bool(meta.get("has_noise")) != has_noise:
                raise ValueError("cache has_noise mismatch")
            if has_noise and noise_arr is not None:
                expected_hash = hashlib.sha256(noise_arr.tobytes()).hexdigest()
                if meta.get("noise_sha256") != expected_hash:
                    raise ValueError("cache noise hash mismatch")
            meta_bounds = np.asarray(meta.get("bounds"), dtype=np.float64)
            if meta_bounds.shape != ba.shape or not np.array_equal(meta_bounds, ba):
                raise ValueError("cache bounds mismatch")

            import torch
            from botorch.models import SingleTaskGP
            from botorch.models.transforms.input import Normalize
            from botorch.models.transforms.outcome import Standardize

            from kalos.core.surrogate import (
                DEVICE,
                DTYPE,
                Surrogate,
                _build_mixed_gp,
                sanitize_bounds,
            )

            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict):
                raise ValueError("loaded weights are not a state_dict mapping")

            Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
            yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
            d = Xt.shape[-1]
            lower, upper = sanitize_bounds(ba)
            box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
            yvar: torch.Tensor | None = None
            if noise_arr is not None:
                yvar = torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

            if cats:
                model: Any = _build_mixed_gp(Xt, yt, box, cats, d, yvar)
            else:
                normalize = Normalize(d=d, bounds=box)
                if yvar is None:
                    model = SingleTaskGP(
                        Xt, yt, input_transform=normalize, outcome_transform=Standardize(m=1)
                    )
                else:
                    model = SingleTaskGP(
                        Xt, yt, train_Yvar=yvar, input_transform=normalize,
                        outcome_transform=Standardize(m=1),
                    )
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            surrogate = Surrogate()
            surrogate.model = model
            surrogate._X = Xt
            surrogate._y = ya.copy()
            surrogate._cat_dims = cats or None
            diag = meta.get("fit_diagnostics")
            surrogate.fit_diagnostics = dict(diag) if isinstance(diag, dict) else {}
            self._touch(meta_path, weights_path)
            return surrogate
        except Exception as exc:
            log.warning("SurrogateCache miss due to corrupt/incompatible entry %s: %s", key, exc)
            return None

    def put(
        self,
        surrogate: Any,
        X_fit: Any,
        y_fit: Any,
        bounds: Any,
        *,
        cat_dims: Sequence[int] | None = None,
        noise: float | np.ndarray | Sequence[float] | None = None,
        restarts: int = 1,
        restart_seed: int = 0,
        check_conditioning: bool = False,
        engine_version: str | None = None,
    ) -> str | None:
        """Persist a fitted ``Surrogate`` atomically and enforce the LRU cap."""
        if getattr(surrogate, "model", None) is None:
            raise ValueError("cannot cache an unfitted Surrogate")
        if self.max_entries <= 0:
            return None
        if engine_version is None:
            from kalos import __version__

            engine_version = __version__

        Xa, ya, ba, cats, noise_arr = _canonical_inputs(X_fit, y_fit, bounds, cat_dims, noise)
        key = self.cache_key(
            Xa, ya, ba, cat_dims=cats, noise=noise_arr, restarts=restarts,
            restart_seed=restart_seed, check_conditioning=check_conditioning,
            engine_version=engine_version,
        )
        weights_path, meta_path = self._paths(key)

        meta = {
            "version": _CACHE_FORMAT_VERSION,
            "key": key,
            "engine_version": str(engine_version),
            "shape": [int(Xa.shape[0]), int(Xa.shape[1])],
            "bounds": ba.tolist(),
            "cat_dims": cats,
            "has_noise": noise_arr is not None,
            "noise_sha256": (
                hashlib.sha256(noise_arr.tobytes()).hexdigest() if noise_arr is not None else None
            ),
            "yvar": noise_arr.tolist() if noise_arr is not None else None,
            "restarts": int(restarts),
            "restart_seed": int(restart_seed),
            "check_conditioning": bool(check_conditioning),
            "fit_diagnostics": _json_safe(dict(getattr(surrogate, "fit_diagnostics", {}) or {})),
        }

        w_tmp: Path | None = None
        m_tmp: Path | None = None
        try:
            import torch

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir, prefix=f".{key}.", suffix=".pt.tmp", delete=False
            ) as wf:
                w_tmp = Path(wf.name)
                torch.save(surrogate.model.state_dict(), wf)
                wf.flush()
            os.replace(w_tmp, weights_path)
            w_tmp = None

            payload = json.dumps(meta, allow_nan=False, separators=(",", ":"))
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.cache_dir,
                prefix=f".{key}.", suffix=".json.tmp", delete=False,
            ) as mf:
                m_tmp = Path(mf.name)
                mf.write(payload)
                mf.flush()
            os.replace(m_tmp, meta_path)
            m_tmp = None

            self._touch(meta_path, weights_path)
            self._evict_if_needed(keep_key=key)
            return key
        except Exception as exc:
            log.warning("SurrogateCache put failed for %s: %s", key, exc)
            for tmp in (w_tmp, m_tmp):
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
            return None

    def _evict_if_needed(self, *, keep_key: str) -> None:
        try:
            entries: list[tuple[int, str, Path]] = []
            for meta_file in self.cache_dir.glob("*.json"):
                if meta_file.name.startswith("."):
                    continue
                try:
                    st = meta_file.stat()
                    entries.append((st.st_mtime_ns, meta_file.stem, meta_file))
                except OSError:
                    continue
            if len(entries) <= self.max_entries:
                return
            entries.sort(key=lambda item: (item[0], item[1]))
            to_remove = len(entries) - self.max_entries
            for _mtime, stem, meta_file in entries:
                if to_remove <= 0:
                    break
                if stem == keep_key:
                    continue
                weights_file = self.cache_dir / f"{stem}.pt"
                for p in (meta_file, weights_file):
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                to_remove -= 1
        except OSError as exc:
            log.warning("SurrogateCache LRU eviction failed: %s", exc)


__all__ = ["SurrogateCache"]
