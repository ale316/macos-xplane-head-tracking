#!/bin/bash
# Installs or updates X-Plane Head Tracker. Everything goes into one folder (delete it to
# uninstall): the code, a private copy of Python and the packages, and the face and hand models.
#
#   One line, no download needed:
#     curl -fsSL https://raw.githubusercontent.com/ale316/macos-xplane-head-tracking/main/install.sh | bash
#   Or from a downloaded copy:
#     bash install.sh
set -euo pipefail

REPO=ale316/macos-xplane-head-tracking
INSTALL_DIR="${HEADTRACK_DIR:-$HOME/X-Plane Head Tracker}"
UV_VERSION=0.11.25
UV_SHA256=5fc334bb25d19806262efd1f6e7d380155c7e817d89bf426df4ba7ae873c9471
PYTHON_VERSION=3.12
MODELS=(  # file, url, sha256
  face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
  hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
)

say() { printf '\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  fail "Sorry, X-Plane Head Tracker needs a Mac with Apple Silicon (M1 or newer)."
fi

# Run from a copy of the code, or fetch the latest code (keeps your settings when updating).
here=$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)
if [ -n "$here" ] && [ -f "$here/headtrack/__init__.py" ]; then
  cd "$here"
else
  say "Downloading X-Plane Head Tracker into $INSTALL_DIR"
  mkdir -p "$INSTALL_DIR"
  cd "$INSTALL_DIR"
  rm -rf headtrack
  curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/main" | tar -xz --strip-components 1
fi

# uv (https://docs.astral.sh/uv/) installs Python and the packages. A pinned, checksummed
# copy is kept in .tools/, so nothing outside this folder is touched.
UV=.tools/uv
if [ ! -x "$UV" ] || [ "$("$UV" --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  say "Downloading uv $UV_VERSION"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz"
  [ "$(sha256 "$tmp/uv.tar.gz")" = "$UV_SHA256" ] || fail "The uv download doesn't match its checksum. Please try again."
  tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
  mkdir -p .tools
  mv "$tmp/uv-aarch64-apple-darwin/uv" "$UV"
  rm -rf "$tmp"
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" UV_NO_CACHE=1

if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ] || ! .venv/bin/python -c "" 2>/dev/null; then
  say "Installing Python $PYTHON_VERSION and packages (a minute or two the first time)"
  rm -rf .venv
  "$UV" venv --quiet --seed --python "$PYTHON_VERSION" --python-preference only-managed .venv
  "$UV" pip install --quiet --python .venv/bin/python -r requirements.txt
  touch .venv/.installed
fi

mkdir -p models
for ((i = 0; i < ${#MODELS[@]}; i += 3)); do
  file=models/${MODELS[i]} url=${MODELS[i + 1]} sum=${MODELS[i + 2]}
  if [ ! -f "$file" ] || [ "$(sha256 "$file")" != "$sum" ]; then
    say "Downloading ${MODELS[i]}"
    curl -fsSL -o "$file.part" "$url"
    [ "$(sha256 "$file.part")" = "$sum" ] || { rm -f "$file.part"; fail "$file doesn't match its checksum. Please try again."; }
    mv "$file.part" "$file"
  fi
done

# A launcher you can double-click in Finder. Downloaded files are quarantined by macOS;
# you've just chosen to install from Terminal, so clear that for this folder.
printf '#!/bin/bash\ncd "$(dirname "$0")" && exec ./run.sh "$@"\n' > "Start Head Tracker.command"
chmod +x "Start Head Tracker.command" run.sh install.sh
xattr -dr com.apple.quarantine . 2>/dev/null || true

if [ -z "${HEADTRACK_FROM_RUN:-}" ]; then
  say "Installed in: $PWD"
  echo "Start it by double-clicking \"Start Head Tracker\" in that folder (opening it in Finder now),"
  echo "or from Terminal with: \"$PWD/run.sh\""
  open -R "Start Head Tracker.command" 2>/dev/null || true
fi
