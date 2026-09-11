# DeepSeek V4 Flash on Modal + OpenCode — v2

Production-oriented repository for `deepseek-ai/DeepSeek-V4-Flash` on Modal with SGLang and an OpenAI-compatible gateway for OpenCode CLI.

## What changed from v1

- Keeps **4 × B200** and tensor parallelism 4.
- Fixes the Modal lifecycle `OSError: [Errno 9] Bad file descriptor` path by giving SGLang an explicit log file descriptor and removing `start_new_session=True`.
- Keeps SGLang private on localhost and exposes only the authenticated gateway.
- Uses the stable OpenCode `provider` schema and `@ai-sdk/openai-compatible`.
- Includes smoke tests and deployment scripts.

## Deploy

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal token new
export DEEPSEEK_MODAL_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
modal secret create deepseek-v4-flash-api DEEPSEEK_MODAL_API_KEY="$DEEPSEEK_MODAL_API_KEY"
modal deploy modal/app.py
```

The app requests `B200:4`. A four-GPU request is below a 10-GPU concurrency allowance; a message saying Modal is acquiring B200 capacity is a capacity/scheduling condition, not proof of a concurrency violation.

## Verify

```bash
modal app list
modal app logs deepseek-v4-flash
```

Wait for:

```text
SGLang is healthy.
Gateway listening on :8000
```

Then:

```bash
export DEEPSEEK_MODAL_BASE_URL='https://YOUR-WORKSPACE--deepseek-v4-flash.modal.run/v1'
./scripts/smoke_test.sh
```

## OpenCode

Stable OpenCode custom providers use `provider`, `@ai-sdk/openai-compatible`, `options.baseURL`, and a model map. The checked-in config uses:

```text
deepseek-modal/deepseek-v4-flash
```

Set the two environment variables and run:

```bash
opencode
```

Then `/models` and select the DeepSeek model, or leave the checked-in `model` setting as the default.

## Troubleshooting

### `OSError: [Errno 9] Bad file descriptor`

This v2 repository replaces the fragile Popen stdio handling used in v1.

### `modal-http: invalid function call` / 404

Do not troubleshoot OpenCode first. Check:

```bash
modal app logs deepseek-v4-flash
```

The Modal Server must successfully start the gateway before `/v1/models` can work.

### B200 scheduling

4 × B200 is a valid four-GPU request. GPU concurrency allowance and physical GPU capacity are separate. If Modal says it is acquiring B200 capacity, the worker has not yet been allocated.

### Security

Use a strong random API key and store it in a Modal Secret. Never commit the key to Git.
