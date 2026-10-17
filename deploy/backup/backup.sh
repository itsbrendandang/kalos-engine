#!/bin/sh
# kalos SQLite backup sidecar loop. Runs inside the `backup` service in
# compose.yaml, sharing the engine's state volume (read-only) plus a
# host-mounted backup directory.
#
# WHY .backup AND NOT `cp`/`cat`/`tar` OF THE LIVE .db FILES
# ------------------------------------------------------------------------
# A file-level copy of a live SQLite database can land mid-write. In
# rollback-journal mode (experiments.db, kalos/store/sqlite_store.py) the .db
# file and its `-journal` are transiently inconsistent; in WAL mode
# (portal.db since schema v2, kalos/portal/migrate.py) committed data lives
# in `-wal` until a checkpoint, so copying the .db file alone silently drops
# recent commits. Either way the copy LOOKS valid (it opens) but is missing or
# corrupting data - the worst kind of backup failure.
#
# `sqlite3 <db> ".backup <dest>"` uses SQLite's Online Backup API: it reads
# committed pages through SQLite itself (WAL included), producing a
# transactionally-consistent snapshot while the engine keeps writing. This is
# correct in both journal modes.
#
# WAL ON A READ-ONLY MOUNT (verified 2026-10-17 with this image, sqlite 3.45.3,
# against a `:ro` named volume and a concurrently writing engine)
# ------------------------------------------------------------------------
# This sidecar mounts the state volume `:ro`, so it cannot create or write
# portal.db's `-shm`. SQLite >= 3.22 still reads a WAL database read-only as
# long as `-wal` and `-shm` already exist, which they do while the engine has
# the database open: `.backup` then works and sees every commit.
# When the engine is STOPPED, its last connection checkpoints and deletes
# `-wal`/`-shm`, and a plain open of the WAL database on the read-only mount
# fails ("unable to open database file"). In that state the main file holds
# everything and nothing is writing it, so `backup_db` retries with
# `?immutable=1`. That fallback is used ONLY when no `-wal` file exists; while
# one exists, immutable would ignore it and miss commits. Every snapshot is
# then checked with `PRAGMA quick_check` before it gets its final name.
#
# The legacy per-tenant analysis cache under `<state>/latest/*.json` is no
# longer written: since schema v2 analyses are rows in portal.db (and so are
# covered by its snapshot). Existing files are kept for the migration and for
# `python -m kalos.portal.migrate down`, so they are still copied
# opportunistically below, not treated as a restore-critical artifact.
set -eu

# Default matches deploy/engine/Containerfile's KALOS_STATE_DIR (/home/kalos/.kalos) -
# see that file's comment for the current state: SqliteStore
# (experiments.db), the runner lock, CampaignStore (portal.db), and the
# `_LATEST` cache all now read KALOS_STATE_DIR at call time, so this only
# needs to track whatever the engine container is actually configured with,
# not hardcode a path those modules ignore. The literal default here still
# matches the engine's, so the two stay in step even if someone runs this
# script with no KALOS_STATE_DIR set at all.
STATE_DIR="${KALOS_STATE_DIR:-/home/kalos/.kalos}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"
INTERVAL="${BACKUP_INTERVAL_SECONDS:-86400}"

log() {
    printf '%s backup: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

backup_db() {
    name="$1"
    src="$STATE_DIR/$name.db"
    if [ ! -f "$src" ]; then
        log "$src not present yet (nothing uploaded/run since last start) - skipping $name"
        return 0
    fi
    dest_dir="$BACKUP_DIR/$name"
    mkdir -p "$dest_dir"
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    dest="$dest_dir/${name}-${ts}.db"
    tmp="$dest.tmp"
    # Online Backup API - safe against a concurrently-writing engine process,
    # unlike a plain file copy. See header comment for why, and for the
    # immutable fallback on a stopped engine's WAL database.
    if ! sqlite3 "$src" ".timeout 5000" ".backup '$tmp'" 2>/dev/null; then
        rm -f "$tmp"
        if [ -e "$src-wal" ] || ! sqlite3 "file:$src?immutable=1" ".backup '$tmp'"; then
            rm -f "$tmp"
            log "FAILED to back up $src - left previous backups untouched"
            return 1
        fi
        log "$name: engine not running (no -wal); snapshot taken with immutable=1"
    fi
    if [ "$(sqlite3 "$tmp" "PRAGMA quick_check" 2>/dev/null)" != "ok" ]; then
        rm -f "$tmp"
        log "FAILED: snapshot of $src did not pass quick_check - left previous backups untouched"
        return 1
    fi
    mv "$tmp" "$dest"
    log "wrote $dest ($(du -h "$dest" | cut -f1))"
}

backup_latest_cache() {
    src="$STATE_DIR/latest"
    [ -d "$src" ] || return 0
    dest_dir="$BACKUP_DIR/latest-cache"
    mkdir -p "$dest_dir"
    # Best-effort, per the header comment - a plain recursive copy is fine
    # here because this cache is not the durable record of anything; the
    # sqlite stores are.
    cp -r "$src" "$dest_dir/$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
}

prune_old() {
    # Applies uniformly under $BACKUP_DIR: any backup artifact (sqlite
    # snapshot or latest-cache directory) older than KEEP_DAYS is removed.
    find "$BACKUP_DIR" -mindepth 2 -maxdepth 2 -mtime "+$KEEP_DAYS" -exec rm -rf {} + 2>/dev/null || true
}

log "starting - state=$STATE_DIR backups=$BACKUP_DIR keep_days=$KEEP_DAYS interval=${INTERVAL}s"

while true; do
    # `|| true`: under `set -e` one failed database used to kill the whole
    # loop (and the other database's backup); the failure is already logged.
    backup_db experiments || true
    backup_db portal || true
    backup_latest_cache
    prune_old
    sleep "$INTERVAL"
done
