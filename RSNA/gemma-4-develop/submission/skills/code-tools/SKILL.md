---
name: code-tools
description: Low-context Python navigation (file outline, go-to-definition including async defs, references, exact symbol source) plus reliable edit/write commands that tolerate quote and backslash escaping and refuse edits that break syntax. Use it for all reading and editing.
---
Call the tool `run_skill_script` with skill_name "code-tools", file_path "scripts/ct.py" and args (a list of strings).
Paths are relative to /workspace. Output is compact; prefer it over reading whole files.

| args | result |
|---|---|
| `["outline", "pkg/mod.py"]` | classes/functions with line ranges. Use before reading a file |
| `["def", "Name"]` or `["def", "Class.method"]` | where it is defined (async defs included) |
| `["refs", "name"]` | every line using it: callers, imports, attributes |
| `["show", "pkg/mod.py", "Class.method"]` or `["show", "pkg/mod.py", "120-160"]` | exact source, no line numbers. Copy OLD text from here |
| `["edit", "pkg/mod.py", OLD, NEW]` | replace the exact text OLD with NEW (5th arg `"all"` replaces every occurrence). Not saved if the result is invalid Python. On a miss it prints the closest text: copy OLD from that |
| `["grep", "PATTERN"]` or `["grep", "PATTERN", "tests"]` | regex search in .py files, at most 40 `file:line: text` hits |
| `["run", "python -m pytest tests/test_x.py -q"]` | run python, pytest or `git status`/`git diff` in /workspace; shows exit code and the end of the output. No search commands |
| `["write", "/tmp/repro.py", CONTENT]` | create or overwrite a file. Scratch files go in /tmp: anything in /workspace is part of your patch |

Write OLD/NEW as plain code, exactly as `show` printed it: `"""` stays `"""`, never `\"\"\"`.
