#!/usr/bin/env bash
# Run the swegemma harness locally: Docker sandbox + Gemma 4 31B on Google AI Studio or OpenRouter.
#   scripts/eval_local.sh <run_name> [task_id ...]     (default tasks: dev_tasks.txt)
#   GOLD=1 scripts/eval_local.sh <run_name>             reference patches instead of the agent
#   PROVIDER=aistudio …                                 own Gemini key instead of paid OpenRouter (free tier is
#                                                       16k input tokens/min: too little for full agent runs)
#   MODEL=google/gemma-4-31b-it:free …                  OpenRouter's shared free pool (often 429s)
# Needs: .env with GEMINI_API_KEY or OPENROUTER_API_KEY, data from scripts/fetch_data.py, image swebench-sandbox:latest.
# Harness console output goes to results/<run_name>/run.log; the model used to results/<run_name>/models.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
# Sandboxes are only cleaned by the harness on a normal finish; also remove them on Ctrl-C/kill.
child=
trap '[ -n "$child" ] && kill "$child" 2>/dev/null; docker ps -q --filter ancestor=swebench-sandbox:latest | xargs -r docker rm -f >/dev/null 2>&1' EXIT INT TERM

name=$1; shift
ids=("$@"); [ ${#ids[@]} -eq 0 ] && ids=($(cat dev_tasks.txt))
mkdir -p "results/$name"

PROVIDER=${PROVIDER:-openrouter}
if [ "$PROVIDER" = aistudio ]; then
  API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
  export MODEL_PROXY_API_KEY="$GEMINI_API_KEY"
  MODEL=${MODEL:-gemma-4-31b-it}
  CONCURRENCY=${CONCURRENCY:-1}  # free tier is per-minute limited
else
  API_BASE=https://openrouter.ai/api/v1
  export MODEL_PROXY_API_KEY="$OPENROUTER_API_KEY"
  MODEL=${MODEL:-google/gemma-4-31b-it}
  case $MODEL in
    *:free) CONCURRENCY=${CONCURRENCY:-1} ;;
    # Pin fp8/bf16 providers that pass tool params (see scripts/swegemma_local.py). Thinking OFF to match the
    # scorer: our bundle's include_thoughts: false becomes enable_thinking=false there (adk_submission
    # resolvers/generation.py), but OpenRouter ignores chat_template_kwargs. THINKING=1 turns it on.
    *) export LLM_EXTRA_BODY=${LLM_EXTRA_BODY:-'{"provider":{"order":["parasail","novita"],"allow_fallbacks":false,"require_parameters":true},"reasoning":{"enabled":'"$([ -n "${THINKING:-}" ] && echo true || echo false)"'}}'} ;;
  esac
fi
# Scorer serves the int4 build on vLLM; these serve other builds, so local scores are indicative.
printf 'models:\n  gemma-4-31b-it-qat-w4a16-ct:\n    path: openai/%s\n    api_base: %s\n' \
  "$MODEL" "$API_BASE" >"results/$name/models.yaml"

.venv/bin/python scripts/swegemma_local.py eval \
  --tasks data/tasks.jsonl --snapshots-dir data/snapshots \
  --submission-dir "${SUBMISSION:-submission}" --results-dir "results/$name" \
  --models-yaml "results/$name/models.yaml" \
  --max-time-minutes "${MAX_MIN:-30}" --concurrency "${CONCURRENCY:-2}" --display quiet \
  ${EXTRA_ARGS:-} --task-ids "${ids[@]}" >"results/$name/run.log" 2>&1 &
child=$!
wait "$child"  # not a foreground child: bash defers traps until it exits, so Ctrl-C/kill would leak sandboxes
cat "results/$name/summary.json"
