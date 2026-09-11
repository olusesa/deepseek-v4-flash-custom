# Operations

1. `modal deploy modal/app.py`
2. `modal app logs deepseek-v4-flash`
3. Wait for `SGLang is healthy.`
4. Set `DEEPSEEK_MODAL_BASE_URL` ending in `/v1`.
5. Run `scripts/smoke_test.sh`.
6. Only after `/models` returns 200, configure OpenCode.

The deployment uses 4 B200 GPUs and TP=4. A capacity wait is distinct from a workspace concurrency limit.
