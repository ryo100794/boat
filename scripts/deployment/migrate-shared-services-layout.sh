#!/usr/bin/env bash
set -Eeuo pipefail

OLD=/workspace/shared-services
PG=/workspace/postgresql
BOAT=/workspace/boat

if pgrep -f '[p]ostgres .*shared-services/postgresql/data' >/dev/null; then
  echo "PostgreSQL must be stopped before layout migration" >&2
  exit 1
fi
if pgrep -f 'boatrace_ai.runtime.postgresql_collector' >/dev/null; then
  echo "Collector must be stopped before layout migration" >&2
  exit 1
fi
if [[ -e "$PG/data" || -e "$PG/runtime" ]]; then
  echo "PostgreSQL target already exists under $PG" >&2
  exit 1
fi

install -d -m 0750 "$PG" "$BOAT/data" "$BOAT/logs" "$BOAT/run"
mv "$OLD/postgresql/runtime" "$PG/runtime"
mv "$OLD/postgresql/data" "$PG/data"
mv "$OLD/postgresql/log" "$PG/log"
mv "$OLD/postgresql/backup" "$PG/backup"
mv "$OLD/postgresql/conf" "$PG/conf"
mv "$OLD/postgresql/migration" "$PG/migration"
if [[ -d "$PG/conf/build-audit" ]]; then
  mv "$PG/conf/build-audit" "$PG/build-audit"
fi

if [[ -d "$OLD/boatrace-runtime/data/raw" ]]; then
  mv "$OLD/boatrace-runtime/data/raw" "$BOAT/data/raw"
fi
if [[ -d "$OLD/boatrace-runtime/log" ]]; then
  mv "$OLD/boatrace-runtime/log" "$BOAT/logs/runtime"
fi

sed -i 's#/workspace/shared-services/postgresql/log#/workspace/postgresql/log#g' \
  "$PG/data/conf.d/shared.conf" \
  "$PG/conf/postgresql-shared.conf"
sed -i 's#/workspace/shared-services/postgresql/conf#/workspace/postgresql/conf#g' \
  "$PG/conf/databases/boat.env" \
  "$PG/conf/databases/boatrace.env"

find "$OLD/postgresql/run" "$OLD/boatrace-runtime/run" -maxdepth 1 -type f -delete 2>/dev/null || true
rmdir "$OLD/postgresql/run" "$OLD/postgresql" 2>/dev/null || true
rmdir "$OLD/boatrace-runtime/data" "$OLD/boatrace-runtime" 2>/dev/null || true

echo "Layout migrated: PostgreSQL=$PG, boatrace runtime=$BOAT"
