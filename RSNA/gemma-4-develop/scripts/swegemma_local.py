"""`swegemma` CLI with the fixes local eval on the public tasks needs. Same args as `swegemma`.

Why: the harness unpacks every small public wheel into site-packages, including released
fastapi/requests/rich/httpx. A package dir in site-packages beats the editable install's
.pth path, so tests ran against the release, not /workspace. On dev_tasks.txt with no
patch at all, 3/10 tasks "resolved" and httpx failed on release-only APIs.
Fix: after every dependency install, delete the task's own package from site-packages and
drop the injection markers so a pooled container re-injects cleanly for the next task.

GOLD=1 replaces the agent with the reference patch: the harness sanity check (expect 100%).
"""

import os
import sys

from adk_eval_core.tracing import SessionTrace
from swegemma import evaluate
from swegemma.cli import main
from swegemma.harness import agent_runner, container_setup, verification

SP = "/usr/local/lib/python3.13/site-packages"
_install = container_setup.install_test_dependencies


def install_test_dependencies(docker, container_id, repo="", **kw):
    _install(docker, container_id, repo, **kw)
    pkg = repo.rsplit("/", 1)[-1].lower()
    if pkg:
        docker.exec(container_id, f"rm -rf {SP}/{pkg} {SP}/.swegemma_base_injected {SP}/.swegemma_whl_*")
    # inline_snapshot (added for newer fastapi tests) auto-loads as a pytest plugin and crashes
    # other repos' sessions, so keep it only where the repo's tests use it.
    docker.exec(container_id, f"grep -rqs inline_snapshot /workspace/tests || rm -rf {SP}/inline_snapshot*")
    # ponytail: injection still takes the newest wheel per package (e.g. starlette without
    # on_startup), so older fastapi commits can't pass even with the gold patch. Such tasks are
    # filtered out of dev_tasks.txt by the gold/no-patch check rather than fixed here.


for mod in (evaluate, agent_runner, verification):
    mod.install_test_dependencies = install_test_dependencies

if os.environ.get("LLM_EXTRA_BODY"):
    # OpenRouter spreads requests over ~14 providers (fp4..bf16, different tool/thinking parsers);
    # unpinned runs saw string-typed tool args and raw "<|channel>thought" replies. Pin providers
    # and enable reasoning so turns behave like the scorer's vLLM (gemma4 tool + reasoning parsers).
    import json

    from google.adk.models.lite_llm import LiteLlm

    _init = LiteLlm.__init__

    def _init_with_body(self, *args, **kwargs):
        kwargs.setdefault("extra_body", json.loads(os.environ["LLM_EXTRA_BODY"]))
        _init(self, *args, **kwargs)

    LiteLlm.__init__ = _init_with_body

# The scorer's vLLM (max_model_len=32768) rejects any request with prompt + max_tokens > 32768, a
# non-retryable error that ends the task. OpenRouter allows 262k, so enforce the limit here after each
# call, using the provider's exact prompt token count. SCORER_CTX_LIMIT=0 disables it.
CTX_LIMIT = int(os.environ.get("SCORER_CTX_LIMIT", "32768"))
if CTX_LIMIT:
    from google.adk.models.lite_llm import LiteLLMClient

    _acompletion = LiteLLMClient.acompletion

    async def _acompletion_with_limit(self, *args, **kwargs):
        response = await _acompletion(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        max_out = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 0
        if usage and usage.prompt_tokens + max_out > CTX_LIMIT:
            raise ValueError(
                f"[scorer emulation] This model's maximum context length is {CTX_LIMIT} tokens. However, you "
                f"requested {max_out} output tokens and your prompt contains {usage.prompt_tokens} input tokens.")
        return response

    LiteLLMClient.acompletion = _acompletion_with_limit

if os.environ.get("GOLD"):
    async def _gold(self, task, *args, **kwargs):
        return task.patch, None, SessionTrace()

    evaluate.Evaluator._run_agent_sandbox = _gold

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
