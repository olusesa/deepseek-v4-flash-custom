#!/usr/bin/env bash
set -euo pipefail
: "${DEEPSEEK_MODAL_BASE_URL:?Set DEEPSEEK_MODAL_BASE_URL including /v1}"
: "${DEEPSEEK_MODAL_API_KEY:?Set DEEPSEEK_MODAL_API_KEY}"
curl --fail-with-body -sS -H "Authorization: Bearer $DEEPSEEK_MODAL_API_KEY" "$DEEPSEEK_MODAL_BASE_URL/models" | python3 -m json.tool
curl --fail-with-body -sS -H "Authorization: Bearer $DEEPSEEK_MODAL_API_KEY" -H 'Content-Type: application/json' "$DEEPSEEK_MODAL_BASE_URL/chat/completions" -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reply with exactly: DeepSeek V4 Flash is online."}],"temperature":0,"max_tokens":32}' | python3 -m json.tool
