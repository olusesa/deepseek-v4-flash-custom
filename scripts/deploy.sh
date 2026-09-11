#!/usr/bin/env bash
set -euo pipefail
: "${DEEPSEEK_MODAL_API_KEY:?Set DEEPSEEK_MODAL_API_KEY}"
modal deploy modal/app.py
