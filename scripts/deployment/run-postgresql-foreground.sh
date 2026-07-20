#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/postgresql
BIN="$ROOT/runtime/bin"
DATA="$ROOT/data"
LOG="$ROOT/log"
SOCKET=/tmp/pgshared
RUN_UID=1999

if [[ ! -x "$BIN/postgres" || ! -f "$DATA/PG_VERSION" ]]; then
  echo "PostgreSQL runtime or cluster is missing under $ROOT" >&2
  exit 1
fi

run_user="$(getent passwd "$RUN_UID" | cut -d: -f1 || true)"
if [[ -z "$run_user" ]]; then
  getent group postgres >/dev/null || groupadd --system postgres
  useradd --system --uid "$RUN_UID" --gid postgres \
    --home-dir "$ROOT" --shell /usr/sbin/nologin postgres
  run_user=postgres
fi

install -d -m 0700 -o "$RUN_UID" "$SOCKET"
install -d -m 0750 "$LOG" "$ROOT/backup"
chown "$RUN_UID" "$DATA" "$LOG" "$ROOT/backup"

if [[ -f "$DATA/postmaster.pid" ]]; then
  pid="$(sed -n '1p' "$DATA/postmaster.pid")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || \
     [[ ! -r "/proc/$pid/cmdline" ]] || \
     ! tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$DATA"; then
    rm -f "$DATA/postmaster.pid"
  fi
fi

exec runuser -u "$run_user" -- "$BIN/postgres" -D "$DATA"
