#!/bin/bash
set -euo pipefail

# Download Proly.app (macOS) into current folder.
# Source: https://github.com/PAIA-Playful-AI-Arena/Proly/releases/tag/1.4.0-beta.1

URL="https://github.com/PAIA-Playful-AI-Arena/Proly/releases/download/1.4.0-beta.1/Proly-darwin-universal-1.4.0-beta.1.zip"
ZIP_NAME="Proly-darwin-universal-1.4.0-beta.1.zip"

if [[ -d "Proly.app" ]]; then
  echo "Proly.app already exists."
  exit 0
fi

if [[ ! -f "$ZIP_NAME" ]]; then
  echo "Downloading $ZIP_NAME ..."
  curl -L "$URL" -o "$ZIP_NAME"
fi

echo "Unzipping $ZIP_NAME ..."
unzip -o "$ZIP_NAME"

echo "Done. Proly.app is ready."
