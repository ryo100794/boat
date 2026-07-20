#!/usr/bin/env bash
set -euo pipefail

raw_dir="${BOATRACE_RAW_DIR:-/workspace/boat/data/raw}"
staging_dir="${BOATRACE_RAW_ARCHIVE_STAGING:-/workspace/boat/data/archive-staging/raw}"
remote="${BOATRACE_RAW_ARCHIVE_REMOTE:-gdrive:workspace/boat/raw/archives}"
config="${RCLONE_CONFIG:-/workspace/google-drive/rclone.conf}"
min_age_minutes="${BOATRACE_RAW_ARCHIVE_MIN_AGE_MINUTES:-30}"
batch_files="${BOATRACE_RAW_ARCHIVE_BATCH_FILES:-5000}"
interval="${BOATRACE_RAW_ARCHIVE_INTERVAL:-600}"
rclone_bin="${RCLONE_BIN:-rclone}"

[[ -f "$config" ]] || { echo "rclone config not found: $config" >&2; exit 2; }
[[ "$min_age_minutes" =~ ^[0-9]+$ ]] || { echo "invalid minimum age: $min_age_minutes" >&2; exit 2; }
[[ "$batch_files" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch size: $batch_files" >&2; exit 2; }
mkdir -p "$raw_dir" "$staging_dir"

upload_archive() {
  local archive="$1"
  local checksum="${archive}.sha256"

  "$rclone_bin" mkdir "$remote" --config "$config"
  "$rclone_bin" copyto "$archive" "$remote/$(basename "$archive")" \
    --config "$config" --drive-chunk-size 64M --retries 5 \
    --low-level-retries 10 --contimeout 30s --timeout 10m
  "$rclone_bin" copyto "$checksum" "$remote/$(basename "$checksum")" \
    --config "$config" --retries 5 --low-level-retries 10 \
    --contimeout 30s --timeout 5m
}

finish_archive() {
  local archive="$1"
  local base="${archive%.tar.zst}"
  local list="${base}.files0"

  upload_archive "$archive"
  if [[ -f "$list" ]]; then
    (cd "$raw_dir" && xargs -0 -r rm -f -- <"$list")
  fi
  find "$raw_dir" -mindepth 1 -depth -type d -empty -delete
  rm -f "$archive" "${archive}.sha256" "$list"
}

archive_batch() {
  local stamp candidates list partial archive checksum count
  stamp="$(date -u +%Y%m%dT%H%M%S%NZ)-$$-$RANDOM"
  candidates="$staging_dir/raw-${stamp}.candidates0"
  list="$staging_dir/raw-${stamp}.files0"
  partial="$staging_dir/raw-${stamp}.tar.zst.partial"
  archive="${partial%.partial}"
  checksum="${archive}.sha256"

  (
    cd "$raw_dir"
    find . -type f -mmin "+$min_age_minutes" -print0 | sort -z >"$candidates"
  )
  head -z -n "$batch_files" "$candidates" >"$list"
  rm -f "$candidates"
  count="$(tr -cd '\0' <"$list" | wc -c)"
  if [[ "$count" -eq 0 ]]; then
    rm -f "$list"
    return 1
  fi

  echo "archiving $count files as $(basename "$archive")"
  tar -C "$raw_dir" --null --files-from "$list" -cf - \
    | zstd -T0 -3 -o "$partial"
  zstd -q -t "$partial"
  mv "$partial" "$archive"
  (cd "$staging_dir" && sha256sum "$(basename "$archive")" >"$(basename "$checksum")")

  finish_archive "$archive"
  echo "uploaded and removed $count archived source files"
}

run_cycle() {
  local archive orphan

  # A finalized archive is a durable retry point after interruption.
  shopt -s nullglob
  for archive in "$staging_dir"/*.tar.zst; do
    [[ -f "${archive}.sha256" ]] || continue
    finish_archive "$archive"
  done
  for orphan in "$staging_dir"/*.partial "$staging_dir"/*.candidates0 "$staging_dir"/*.files0; do
    [[ -e "$orphan" ]] && rm -f "$orphan"
  done
  shopt -u nullglob

  while archive_batch; do
    :
  done
}

while true; do
  run_cycle
  [[ "${BOATRACE_RAW_ARCHIVE_ONCE:-0}" == "1" ]] && exit 0
  sleep "$interval"
done
