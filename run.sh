#!/bin/bash
# Starts the head tracker. The first run sets everything up inside this folder (a Python
# environment, the packages and the face model); later runs start straight away.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=models/face_landmarker.task
MODEL_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MODEL_SHA256=64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "Sorry, this needs a Mac with Apple Silicon (M1 or newer)." >&2
  exit 1
fi

find_python() {
  for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null &&
       "$py" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null; then
      command -v "$py"
      return
    fi
  done
  return 1
}

if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "Setting up (first run only, takes a minute or two)..."
  rm -rf .venv
  if command -v uv >/dev/null; then
    uv venv --quiet --seed --python 3.12 .venv  # uv downloads a suitable Python if needed
    uv pip install --quiet --python .venv/bin/python -r requirements.txt
  elif PY=$(find_python); then
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
  else
    echo "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/macos/" >&2
    echo "or with Homebrew (brew install python@3.12), then run this again." >&2
    exit 1
  fi
  touch .venv/.installed
fi

if [ ! -f "$MODEL" ] || [ "$(shasum -a 256 "$MODEL" | cut -d' ' -f1)" != "$MODEL_SHA256" ]; then
  echo "Downloading the face model (3.6 MB)..."
  mkdir -p models
  curl -fsSL -o "$MODEL.part" "$MODEL_URL"
  if [ "$(shasum -a 256 "$MODEL.part" | cut -d' ' -f1)" != "$MODEL_SHA256" ]; then
    rm -f "$MODEL.part"
    echo "The downloaded face model doesn't match its checksum. Please try again." >&2
    exit 1
  fi
  mv "$MODEL.part" "$MODEL"
fi

exec .venv/bin/python -m headtrack "$@"
