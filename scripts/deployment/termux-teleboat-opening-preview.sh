#!/data/data/com.termux/files/usr/bin/bash
set -Eeuo pipefail

readonly TERMUX_PREFIX="${TERMUX_PREFIX:-/data/data/com.termux/files/usr}"
readonly DATE_BIN="$TERMUX_PREFIX/bin/date"
readonly PROOT_DISTRO="$TERMUX_PREFIX/bin/proot-distro"
readonly WAKE_LOCK="$TERMUX_PREFIX/bin/termux-wake-lock"
readonly WAKE_UNLOCK="$TERMUX_PREFIX/bin/termux-wake-unlock"

readonly APP_DIR="/root/boat"
readonly PYTHON="$APP_DIR/.venv/bin/python"
readonly PREVIEW_SCRIPT="$APP_DIR/scripts/teleboat_opening_preview.py"
readonly JOURNAL="$APP_DIR/data/teleboat_vote_journal.jsonl"
readonly SECRET="$APP_DIR/.secrets/teleboat-login.json"
readonly DASHBOARD_URL="https://unr72rnwxtcmvv-10001.proxy.runpod.net/"

current_date="$(TZ=Asia/Tokyo "$DATE_BIN" +%F)"
readonly TARGET_DATE="${TELEBOAT_PREVIEW_DATE:-$current_date}"
readonly OUTPUT="$APP_DIR/data/teleboat-opening-preview-$TARGET_DATE.json"
readonly LOG="$APP_DIR/data/teleboat-opening-preview-$TARGET_DATE.log"

[[ "$current_date" == "$TARGET_DATE" ]] || exit 0

# The Python output is atomically written. Only an exact success for this date
# suppresses another attempt; malformed and failed output remains retryable.
if "$PROOT_DISTRO" login ubuntu -- "$PYTHON" -c \
  'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8")); raise SystemExit(0 if d.get("target_date") == sys.argv[2] and d.get("status") == "success" else 1)' \
  "$OUTPUT" "$TARGET_DATE" >/dev/null 2>&1; then
  exit 0
fi

"$WAKE_LOCK"
wake_lock_held=1
release_wake_lock() {
  if [[ "${wake_lock_held:-0}" == 1 ]]; then
    "$WAKE_UNLOCK" >/dev/null 2>&1 || true
    wake_lock_held=0
  fi
}
trap release_wake_lock EXIT HUP INT TERM

# Redirection happens inside Ubuntu, so the append-only log is under
# /root/boat/data. Secret values are loaded only by the Python process.
"$PROOT_DISTRO" login ubuntu -- /bin/bash -c '
  app_dir=$1
  log=$2
  shift 2
  cd "$app_dir"
  /bin/printf "%s opening-preview start\n" "$(/bin/date --iso-8601=seconds)" >>"$log"
  set +e
  "$@" >>"$log" 2>&1
  rc=$?
  set -e
  /bin/printf "%s opening-preview exit=%s\n" "$(/bin/date --iso-8601=seconds)" "$rc" >>"$log"
  exit "$rc"
' termux-teleboat-opening-preview "$APP_DIR" "$LOG" \
  "$PYTHON" "$PREVIEW_SCRIPT" \
  --date "$TARGET_DATE" \
  --dashboard-url "$DASHBOARD_URL" \
  --poll-seconds 30 \
  --timeout-seconds 840 \
  --output "$OUTPUT" \
  --journal-path "$JOURNAL" \
  --secret-path "$SECRET"
