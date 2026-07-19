#!/usr/bin/env bash
set -euo pipefail

PYTHON="${1:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
EXT_SUFFIX="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUTPUT="${ROOT}/src/boatrace_ai/_fast_boat_math${EXT_SUFFIX}"

"${CC:-gcc}" -O3 -march=native -fPIC -shared \
  -I"${INCLUDE}" \
  "${ROOT}/native/fast_boat_math.c" \
  -o "${OUTPUT}"

echo "${OUTPUT}"
