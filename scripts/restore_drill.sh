#!/bin/sh
# kalos backup restore drill - restores the NEWEST backups into a brand-new,
# scratch Docker volume and proves they open the way the engine will open them.
#
# It never touches the live `kalos-state` volume and never stops the engine:
# the drill is safe to run on the production host, and it is meant to be run
# periodically (RUNBOOK.md, "Backup / restore drill") - a backup nobody has
# restored is a hope, not a backup.
#
# What it checks, in order:
#   1. the newest experiments and portal snapshots exist under BACKUP_DIR;
#   2. they are copied into a FRESH named volume, owned the way the engine
#      expects (uid/gid 10001) - the VOLUME ROOT too, not just the .db files:
#      SQLite must create `-wal` and `-shm` next to portal.db, and a root-owned
#      directory makes that fail with "unable to open database file";
#   3. as the engine's own user (uid 10001), with the backup image's sqlite3,
#      each restored file passes `PRAGMA quick_check` and can be switched to
#      WAL mode (which creates -wal/-shm) and read back.
#
# No secrets are read or printed: the drill uses no tokens, salts or env files.
#
# Usage (from the repo root, after building the backup image - see RUNBOOK.md):
#   docker build -f deploy/backup/Containerfile -t kalos-backup:local deploy/backup
#   scripts/restore_drill.sh                 # BACKUP_DIR defaults to deploy/backups
#   BACKUP_DIR=/path/to/backups scripts/restore_drill.sh
#
# Environment (all optional):
#   BACKUP_DIR          host directory holding experiments/ and portal/ (default deploy/backups)
#   KALOS_BACKUP_IMAGE  image with sqlite3 and a shell (default kalos-backup:local)
#   KALOS_DRILL_IMAGE   image used to copy files into the volume (default alpine:3.20)
#   KALOS_DRILL_KEEP=1  keep the scratch volume afterwards for inspection
#
# Exit status: 0 = restore verified; non-zero = the drill failed (the reason is
# printed on stderr). The scratch volume is removed unless KALOS_DRILL_KEEP=1.
set -eu

ENGINE_UID=10001
BACKUP_DIR="${BACKUP_DIR:-deploy/backups}"
BACKUP_IMAGE="${KALOS_BACKUP_IMAGE:-kalos-backup:local}"
DRILL_IMAGE="${KALOS_DRILL_IMAGE:-alpine:3.20}"
KEEP="${KALOS_DRILL_KEEP:-0}"

die() {
    printf 'restore drill: FAILED: %s\n' "$1" >&2
    exit 1
}

log() {
    printf 'restore drill: %s\n' "$1"
}

command -v docker >/dev/null 2>&1 || die "docker is not on PATH"

# 1. newest backups ------------------------------------------------------------
[ -d "$BACKUP_DIR" ] || die "backup directory $BACKUP_DIR does not exist"
exp_src="$(ls -t "$BACKUP_DIR"/experiments/*.db 2>/dev/null | head -n 1 || true)"
portal_src="$(ls -t "$BACKUP_DIR"/portal/*.db 2>/dev/null | head -n 1 || true)"
[ -n "$exp_src" ] || die "no experiments snapshot under $BACKUP_DIR/experiments/"
[ -n "$portal_src" ] || die "no portal snapshot under $BACKUP_DIR/portal/"
log "newest experiments snapshot: $(basename "$exp_src")"
log "newest portal snapshot:      $(basename "$portal_src")"

# 2. fresh scratch volume, restored and owned for the engine user -------------
vol="kalos-restore-drill-$(date -u +%Y%m%dT%H%M%SZ)-$$"
case "$vol" in
    kalos-state|kalos-state-*) die "refusing to use the live volume name" ;;
esac
cleanup() {
    if [ "$KEEP" != "1" ]; then
        docker volume rm "$vol" >/dev/null 2>&1 || true
    else
        log "keeping scratch volume $vol (KALOS_DRILL_KEEP=1)"
    fi
}
trap cleanup EXIT

docker volume create "$vol" >/dev/null
log "scratch volume $vol created"

# The copy runs as root (it has to write into a fresh volume), then hands the
# VOLUME ROOT and both files to the engine's uid. Chowning only the .db files
# is the bug this drill exists to catch: SQLite then cannot create -wal/-shm.
docker run --rm \
    -v "$vol":/state \
    -v "$(cd "$BACKUP_DIR" && pwd)":/backups:ro \
    "$DRILL_IMAGE" sh -c "
        set -eu
        cp '/backups/experiments/$(basename "$exp_src")' /state/experiments.db
        cp '/backups/portal/$(basename "$portal_src")' /state/portal.db
        chown ${ENGINE_UID}:${ENGINE_UID} /state
        chmod 0755 /state
        chown ${ENGINE_UID}:${ENGINE_UID} /state/experiments.db /state/portal.db
    " >/dev/null || die "could not copy the snapshots into the scratch volume"
log "snapshots copied; volume root and files owned by uid $ENGINE_UID"

# 3. verify as the engine's own user ------------------------------------------
# --entrypoint sh: the backup image's entrypoint is the backup loop, not a shell.
docker run --rm --user "$ENGINE_UID:$ENGINE_UID" \
    --entrypoint sh \
    -v "$vol":/state \
    "$BACKUP_IMAGE" -c '
        set -eu
        for db in /state/experiments.db /state/portal.db; do
            result="$(sqlite3 "$db" "PRAGMA quick_check")"
            [ "$result" = "ok" ] || { echo "quick_check failed for $db: $result" >&2; exit 1; }
        done
        # Switching to WAL creates -wal/-shm beside portal.db; this is the step
        # that fails on a root-owned volume root. The -wal/-shm files exist only
        # while the connection is open (the last close removes them), so the
        # existence check runs inside the same sqlite3 session.
        out="$(sqlite3 /state/portal.db "PRAGMA journal_mode=WAL" \
            "CREATE TABLE IF NOT EXISTS restore_drill_probe (x INTEGER); INSERT INTO restore_drill_probe VALUES (1);" \
            ".system test -f /state/portal.db-wal && test -f /state/portal.db-shm && echo wal-files-ok")"
        echo "$out" | grep -qx wal || { echo "portal.db did not enter WAL mode" >&2; exit 1; }
        echo "$out" | grep -qx wal-files-ok || { echo "portal.db could not create -wal/-shm beside it" >&2; exit 1; }
        sqlite3 /state/experiments.db "SELECT COUNT(*) FROM sqlite_master" >/dev/null
    ' || die "restored databases failed verification as uid $ENGINE_UID (see the message above)"

log "PASSED: both snapshots restore, pass quick_check, and open read-write as uid $ENGINE_UID"
