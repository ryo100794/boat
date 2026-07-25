#!/usr/bin/env bash
set -Eeuo pipefail

app_root="${BOATRACE_APP_ROOT:-/workspace/boat}"
model_dir="${BOATRACE_MODEL_DIR:-$app_root/data/models}"
remote_root="${BOATRACE_MODEL_ARCHIVE_REMOTE:-gdrive:workspace/boat/models/cache-archives}"
config="${RCLONE_CONFIG:-/workspace/google-drive/rclone.conf}"
rclone_bin="${RCLONE_BIN:-rclone}"

[[ -f "$config" ]] || { echo "rclone config not found: $config" >&2; exit 2; }
[[ "$#" -gt 0 ]] || { echo "at least one model cache path is required" >&2; exit 2; }

model_dir="$(realpath -e "$model_dir")"
lock_dir="$app_root/data/archive-staging"
mkdir -p "$lock_dir"
exec 9>"$lock_dir/model-cache-archive.lock"
flock -w 300 9 || { echo "model cache archive lock timeout" >&2; exit 75; }

for requested in "$@"; do
  if [[ ! -e "$requested" && -f "${requested}.gdrive.json" ]]; then
    echo "already archived: $requested"
    continue
  fi
  source="$(realpath -e "$requested")"
  case "$source" in
    "$model_dir"/*) ;;
    *) echo "model cache path is outside model directory: $requested" >&2; exit 2 ;;
  esac
  [[ -f "$source" && ! -L "$source" ]] || {
    echo "model cache path must be a regular file: $requested" >&2
    exit 2
  }

  relative="${source#"$model_dir"/}"
  local_md5="$("$rclone_bin" md5sum "$source" --config "$config" | awk 'NR == 1 {print $1}')"
  [[ -n "$local_md5" ]] || { echo "failed to hash $relative" >&2; exit 1; }
  remote="$remote_root/$local_md5/$relative"
  marker="${source}.gdrive.json"
  marker_tmp="${marker}.tmp"
  bytes="$(stat -c %s "$source")"

  "$rclone_bin" copyto "$source" "$remote" \
    --config "$config" --drive-chunk-size 128M --transfers 1 \
    --retries 5 --low-level-retries 10 --contimeout 30s --timeout 30m
  remote_md5="$("$rclone_bin" md5sum "$remote" --config "$config" | awk 'NR == 1 {print $1}')"
  if [[ "$local_md5" != "$remote_md5" ]]; then
    echo "remote model cache verification failed: $relative" >&2
    exit 1
  fi

  printf '{\n  "archived_at": "%s",\n  "bytes": %s,\n  "md5": "%s",\n  "remote": "%s",\n  "source": "%s"\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$bytes" "$local_md5" \
    "$remote" "$relative" >"$marker_tmp"
  mv "$marker_tmp" "$marker"
  rm -f -- "$source"
  echo "verified and archived $relative ($bytes bytes) to $remote"
done
