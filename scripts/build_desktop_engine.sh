#!/usr/bin/env bash
# Build a self-contained Kalos engine for the desktop app (arm64 macOS).
#
#   scripts/build_desktop_engine.sh [OUT_DIR]      (default: dist/kalos-engine)
#
# Output (docs/DESKTOP.md, "Packaged engine"):
#   OUT_DIR/kalos-engine     launcher: exec python -I -m kalos.portal "$@"
#   OUT_DIR/python/          a relocatable CPython (python-build-standalone, the
#                            build uv manages) with the locked dependencies and
#                            the kalos package installed into its site-packages
#   OUT_DIR/BUILD_INFO.txt   versions, git commit, size, smoke-test result
#
# Approach: python-build-standalone + site-packages, not PyInstaller. torch and
# its dylibs are used exactly as the wheels ship them (nothing is re-linked or
# frozen), the interpreter is relocatable, and `-I` keeps the user's PYTHON*
# variables and user site-packages out. Dependencies come from uv.lock
# (`uv export --frozen`, with hashes, installed with --require-hashes), so the
# bundle runs the same torch/BoTorch build the tests and benchmarks use. The
# interpreter is pinned to a full version (KALOS_BUNDLE_PYTHON overrides it).
#
# Refuses (exit 2) to replace an OUT_DIR that is not empty and holds no
# BUILD_INFO.txt (i.e. that is not an earlier bundle). Fails (exit 1) unless the
# smoke test sees 401 without the token and exit 0 after SIGTERM.
#
# Needs: macOS on arm64, uv, network access the first time (wheels, and the
# interpreter if uv has none). Writes only OUT_DIR and a temp dir; never touches
# ~/.kalos. No code signature or notarisation of its own: the shell's .app covers it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
OUT="${1:-$ROOT/dist/kalos-engine}"
PY_VERSION="${KALOS_BUNDLE_PYTHON:-3.12.5}"
PY_MINOR="$(echo "$PY_VERSION" | cut -d. -f1-2)"

# OUT is deleted and rebuilt below: only ever an empty folder or an earlier bundle.
if [ -e "$OUT" ] && { [ ! -d "$OUT" ] || { [ -n "$(ls -A "$OUT")" ] && [ ! -f "$OUT/BUILD_INFO.txt" ]; }; }; then
  echo "build_desktop_engine: $OUT exists and is not an earlier bundle (no BUILD_INFO.txt); refusing to delete it" >&2
  exit 2
fi

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "build_desktop_engine: this script builds for arm64 macOS only (found $(uname -s) $(uname -m))" >&2
  exit 2
fi
command -v uv >/dev/null || { echo "build_desktop_engine: uv is required (brew install uv)" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/kalos-engine-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
t_start=$(date +%s)

# 1. A relocatable interpreter. uv's managed CPython is python-build-standalone.
PY_SRC="$(cd "$WORK" && uv python find --managed-python --no-project "$PY_VERSION" 2>/dev/null || true)"
if [ -z "$PY_SRC" ]; then
  uv python install --install-dir "$WORK/pythons" "$PY_VERSION"
  PY_SRC="$(ls "$WORK"/pythons/cpython-"$PY_VERSION"*/bin/python3 | head -1)"
fi
PY_HOME="$(cd "$(dirname "$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PY_SRC")")/.." && pwd -P)"
echo "interpreter: $PY_HOME"
PY_FOUND="$("$PY_SRC" -I -c 'import platform; print(platform.python_version())')"
if [ "$PY_FOUND" != "$PY_VERSION" ]; then
  echo "build_desktop_engine: found Python $PY_FOUND, need exactly $PY_VERSION" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cp -R "$PY_HOME" "$OUT/python"
PY="$OUT/python/bin/python3"
# The copy is ours: let pip-style installs write to its site-packages.
find "$OUT/python/lib" -maxdepth 2 -name EXTERNALLY-MANAGED -delete

# 2. Locked dependencies, then the engine itself. Extras: ml + portal, plus
#    `assist` (the local assistant: BM25 retrieval + an Ollama client) when this
#    checkout defines it. `assist-embed` (sentence-transformers) is not bundled:
#    the BM25 path works offline. Override with KALOS_BUNDLE_EXTRAS="ml portal ...".
EXTRA_ARGS=()
for extra in ${KALOS_BUNDLE_EXTRAS:-ml portal assist}; do
  if "$PY" -I -c 'import sys, tomllib; sys.exit(0 if sys.argv[2] in tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"] else 1)' "$ROOT/pyproject.toml" "$extra"; then
    EXTRA_ARGS+=(--extra "$extra")
  else
    echo "build_desktop_engine: extra '$extra' is not defined in pyproject.toml; skipped" >&2
  fi
done
echo "extras: ${EXTRA_ARGS[*]}"
(cd "$ROOT" && uv export --frozen --no-dev "${EXTRA_ARGS[@]}" --no-emit-project \
  --format requirements-txt -o "$WORK/requirements.txt" >/dev/null)
uv pip install --python "$PY" --break-system-packages --quiet --require-hashes \
  -r "$WORK/requirements.txt"
(cd "$ROOT" && uv build --wheel --quiet -o "$WORK/wheel" >/dev/null)
uv pip install --python "$PY" --break-system-packages --quiet --no-deps "$WORK"/wheel/kalos-*.whl

# 3. Trim what an engine never runs: headers, static libs, the CPython test
#    suite, Tk, pip. torch/testing stays (torch imports it).
SP="$("$PY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$OUT/python/include" "$OUT/python/share" \
  "$OUT/python/lib/python$PY_MINOR/test" "$OUT/python/lib/python$PY_MINOR/idlelib" \
  "$OUT/python/lib/python$PY_MINOR/tkinter" "$OUT/python/lib/python$PY_MINOR/turtledemo" \
  "$OUT/python/lib/python$PY_MINOR/ensurepip" "$OUT/python/lib/python$PY_MINOR/lib2to3" \
  "$SP/torch/include" "$SP/torch/share/cmake" "$SP"/pip "$SP"/pip-*.dist-info
find "$OUT/python" -name "*.a" -delete
find "$OUT/python" -name "__pycache__" -type d -prune -exec rm -rf {} +

# 4. Byte-compile once, so the first launch does not write .pyc files into the
#    bundle (it may be read-only inside a signed .app).
"$PY" -I -m compileall -q -j 0 "$OUT/python/lib/python$PY_MINOR" >/dev/null 2>&1 || true

# 5. The launcher.
cat > "$OUT/kalos-engine" <<'LAUNCH'
#!/bin/sh
# Kalos engine launcher (scripts/build_desktop_engine.sh). Pass --desktop and the
# environment described in docs/DESKTOP.md.
HERE="$(cd "$(dirname "$0")" && pwd -P)"
exec "$HERE/python/bin/python3" -I -m kalos.portal "$@"
LAUNCH
chmod +x "$OUT/kalos-engine"

# 6. Smoke test: start it in desktop mode on a free port with a temp state dir,
#    wait for /api/health, stop it with SIGTERM. Measures cold start.
SMOKE_STATE="$WORK/state"
SECRET="$("$PY" -I -c 'import secrets; print(secrets.token_hex(24))')"
TOKENS="$("$PY" -I -c 'import hashlib,json,sys; print(json.dumps([{"token_sha256": hashlib.sha256(sys.argv[1].encode()).hexdigest(), "subject": "build-smoke", "tenant": "local", "scopes": ["read", "write"]}]))' "$SECRET")"
PORT="$("$PY" -I -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
t0=$("$PY" -I -c 'import time; print(time.time())')
env -i HOME="$WORK" PATH=/usr/bin:/bin KALOS_DESKTOP=1 KALOS_HOST=127.0.0.1 KALOS_PORT="$PORT" \
  KALOS_STATE_DIR="$SMOKE_STATE" KALOS_AUTH_TOKENS="$TOKENS" KALOS_JOBS_ENABLED=0 \
  KALOS_CORS_ORIGINS="http://127.0.0.1:1" \
  "$OUT/kalos-engine" >"$WORK/smoke.log" 2>&1 &
ENGINE_PID=$!
HEALTH=""
for _ in $(seq 1 600); do
  if HEALTH="$(curl -sf -H "Authorization: Bearer $SECRET" "http://127.0.0.1:$PORT/api/health")"; then break; fi
  kill -0 "$ENGINE_PID" 2>/dev/null || break
  sleep 0.1
done
t1=$("$PY" -I -c 'import time; print(time.time())')
UNAUTH="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/health" || true)"
kill -TERM "$ENGINE_PID" 2>/dev/null || true
if wait "$ENGINE_PID" 2>/dev/null; then EXIT_CODE=0; else EXIT_CODE=$?; fi
if [ -z "$HEALTH" ] || [ "$UNAUTH" != "401" ] || [ "$EXIT_CODE" != "0" ]; then
  HEALTH_SEEN="none"; [ -n "$HEALTH" ] && HEALTH_SEEN="ok"
  echo "build_desktop_engine: smoke test FAILED (health $HEALTH_SEEN," \
    "without token HTTP $UNAUTH, want 401; SIGTERM exit $EXIT_CODE, want 0); engine log:" >&2
  cat "$WORK/smoke.log" >&2
  exit 1
fi
COLD_S="$("$PY" -I -c "print(round($t1 - $t0, 2))")"
SIZE="$(du -sh "$OUT" | cut -f1)"

{
  echo "kalos engine bundle"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(sw_vers -productVersion 2>/dev/null) $(uname -m)"
  echo "git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo ' (dirty)')"
  echo "python: $("$PY" -I -c 'import platform; print(platform.python_version())')"
  echo "extras: ${EXTRA_ARGS[*]}"
  echo "health: $HEALTH"
  echo "size: $SIZE"
  echo "smoke: start to first /api/health ${COLD_S} s (first launch of a fresh bundle), without token HTTP $UNAUTH, SIGTERM exit $EXIT_CODE"
  echo "build time: $(( $(date +%s) - t_start )) s"
} > "$OUT/BUILD_INFO.txt"
cat "$OUT/BUILD_INFO.txt"
