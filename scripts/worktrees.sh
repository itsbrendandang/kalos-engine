#!/usr/bin/env bash
# List every worktree of this repository with what could be lost or cleaned up:
# uncommitted files, commits not on any remote, a remote branch that is gone, or a
# branch already merged into origin/main. Read-only: it never deletes anything.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch -q --prune origin || echo "warning: fetch failed; remote state may be stale" >&2
printf "%-40s %-38s %s\n" WORKTREE BRANCH STATE
git worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r wt; do
  [ -d "$wt" ] || { printf "%-40s %-38s %s\n" "$(basename "$wt")" "-" "missing (git worktree prune)"; continue; }
  br=$(git -C "$wt" symbolic-ref -q --short HEAD || echo "(detached)")
  state=()
  dirty=$(git -C "$wt" status --porcelain | wc -l | tr -d ' ')
  [ "$dirty" != 0 ] && state+=("UNCOMMITTED:$dirty")
  if [ "$br" != "(detached)" ]; then
    up=$(git -C "$wt" rev-parse -q --abbrev-ref "$br@{upstream}" 2>/dev/null || true)
    if [ -z "$up" ]; then state+=("no-upstream")
    elif ! git rev-parse -q --verify "$up" >/dev/null; then state+=("remote-gone")
    fi
    ahead=$(git rev-list --count "$br" --not --remotes 2>/dev/null || echo 0)
    [ "$ahead" != 0 ] && state+=("UNPUSHED:$ahead")
    own=$(git rev-list --count "origin/main..$br" 2>/dev/null || echo 0)
    if [ "$br" != main ] && [ "$own" != 0 ]; then
      # Squash merges change every commit id, so ask GitHub first; git cherry only
      # recognises commits applied one by one.
      if command -v gh >/dev/null && [ -n "$(gh pr list --head "$br" --state merged --json number -q '.[].number' 2>/dev/null)" ]; then
        state+=("merged")
      elif [ -z "$(git cherry origin/main "$br" 2>/dev/null | grep '^+' || true)" ]; then
        state+=("merged")
      fi
    fi
  fi
  printf "%-40s %-38s %s\n" "$(basename "$wt")" "$br" "${state[*]:-clean}"
done
