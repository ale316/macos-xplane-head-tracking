#!/bin/bash
# Starts X-Plane Head Tracker, installing it first if needed.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .venv/.installed ] || ! .venv/bin/python -c "" 2>/dev/null || [ requirements.txt -nt .venv/.installed ] \
   || [ ! -f models/face_landmarker.task ]; then
  HEADTRACK_FROM_RUN=1 bash install.sh
fi
exec .venv/bin/python -m headtrack "$@"
