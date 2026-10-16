`llama-swap-config-schema.json` is llama-swap's `config-schema.json`, copied
unchanged from https://github.com/mostlygeek/llama-swap at commit
`a2796fb0306215c6883454a8707dfb22d25c3068` (release v265, 2026-10-14).
llama-swap is MIT licensed (`LICENSE-llama-swap.md`), which allows
redistributing this copy with its notice.

Refresh it when upgrading llama-swap:

```bash
curl -fsSL https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json \
  -o docker/spark-stack/model-hub/model_hub/vendor/llama-swap-config-schema.json
```
