"""Check for submission/skills/code-tools/scripts/ct.py:  python3 scripts/test_code_tools.py  (BUNDLE=variants/x to test another copy)

The edit cases replay the escaping slips Gemma made in the v1 trace of fastapi_15588.
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # a .pyc inside submission/ is rejected by the validator

spec = importlib.util.spec_from_file_location("ct", Path(__file__).parent.parent / os.environ.get("BUNDLE", "submission") / "skills/code-tools/scripts/ct.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

SRC = '''class ServerSentEvent:
    def check(self, data: str) -> str:
        """
        Must not contain null (`\\\\0`) characters.
        """
        return data


async def serialize(value):
    return value
'''


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ct.main(list(argv))
    return buf.getvalue()


with tempfile.TemporaryDirectory() as root:
    ct.ROOT = root
    f = Path(root, "pkg/sse.py")
    f.parent.mkdir()

    # 1. Gemma's \"\"\" for """ (old AND new escaped the same way) -> normalized, file stays valid
    f.write_text(SRC)
    r = run("edit", "pkg/sse.py", '        \\"\\"\\"\n        Must not', '        \\"\\"\\"\n        Must never')
    assert r.startswith("OK") and "normalized" in r, r
    assert '"""\n        Must never' in f.read_text() and '\\"' not in f.read_text()

    # 2. halved backslash: file has \\0, Gemma sends \0
    f.write_text(SRC)
    r = run("edit", "pkg/sse.py", "(`\\0`) characters.", "(`\\0`) bytes.")
    assert r.startswith("OK"), r
    assert "(`\\\\0`) bytes." in f.read_text()

    # 3. an edit that would break syntax is refused and the file is untouched
    f.write_text(SRC)
    r = run("edit", "pkg/sse.py", "return data", "return (data")
    assert r.startswith("ERROR: not saved") and f.read_text() == SRC, r

    # 4. miss -> closest text shown
    r = run("edit", "pkg/sse.py", "return dataz", "x")
    assert "Closest text is lines 6-6" in r and "return data" in r, r

    # 5. navigation: async defs included, qualnames, exact show
    assert "L9-10 async def serialize(value)" in run("outline", "pkg/sse.py")
    assert "pkg/sse.py:2-6  ServerSentEvent.check" in run("def", "check")
    assert "pkg/sse.py:9-10  serialize" in run("def", "serialize")
    assert "lines 2-6" in run("show", "pkg/sse.py", "ServerSentEvent.check")
    assert run("show", "pkg/sse.py", "9-10").endswith("    return value\n")
    assert "pkg/sse.py:6: return data" in run("refs", "data")

    # 6. write: escaped content is fixed if that makes valid Python; /tmp-style note
    r = run("write", "pkg/new.py", 'x = \\"a\\"\n')
    assert Path(root, "pkg/new.py").read_text() == 'x = "a"\n' and "WILL be in your patch" in r, r

    # 7. run: executes python/pytest/git only; search commands are refused (exploration -> code_analyzer)
    r = run("run", "python -c 'print(6*7)'")
    assert r.startswith("exit code 0") and "42" in r, r
    for bad in ("grep -rn x .", "cat pkg/sse.py", "python -c 1 | grep x", "git log -p"):
        assert run("run", bad).startswith("ERROR: `run` only runs"), bad

    # 8. grep: capped regex search with file:line hits, optional dir filter, bad regex -> literal
    f.write_text(SRC)
    assert "pkg/sse.py:6: return data" in run("grep", r"return\s+data")
    assert run("grep", "return data", "tests").startswith("no match")
    assert "pkg/sse.py:2:" in run("grep", "check(self")

    # 9. hidden files (ADK writes .adk_exec_*.py into /workspace, containing the grep pattern) are ignored
    Path(root, ".adk_exec_x.py").write_text("sys.argv = ['ct.py', 'grep', 'zzqq']\n")
    assert run("grep", "zzqq").startswith("no match")

    # 10. test files in the repo are refused (a changed test file makes the hidden test_patch fail to apply)
    Path(root, "tests").mkdir()
    Path(root, "tests/test_x.py").write_text("assert 1\n")
    assert run("edit", "tests/test_x.py", "assert 1", "assert 2").startswith("ERROR: refused")
    assert run("write", "tests/test_new.py", "x = 1\n").startswith("ERROR: refused")
    assert run("write", "conftest.py", "x = 1\n").startswith("ERROR: refused")
    assert Path(root, "tests/test_x.py").read_text() == "assert 1\n"
    assert run("edit", "pkg/sse.py", "return data", "return data ").startswith("OK")  # source still editable

    # 11. tool parameters pasted into args are skipped
    assert "L9-10 async def serialize(value)" in run("run_skill_script", "code-tools", "scripts/ct.py", "outline", "pkg/sse.py")

    # 12. docs: grep searches markdown too, outline lists markdown headings with section ranges
    Path(root, "docs").mkdir()
    Path(root, "docs/guide.md").write_text("# Guide\n\nintro\n\n## Streaming\n\nuse yield\n```\n# not a heading\n```\n## Other\nx\n")
    if hasattr(ct, "md_outline"):
        assert "docs/guide.md:7: use yield" in run("grep", "use yield")  # only docs match -> docs shown
        Path(root, "pkg/y.py").write_text("x = 'use yield'\n")
        g = run("grep", "use yield")
        assert g.startswith("[source]\npkg/y.py:1:") and "+ 1 matches in docs/other files" in g, g
        assert "docs/guide.md:7" in run("grep", "use yield", "docs")
        o = run("outline", "docs/guide.md")
        assert "L1-12 Guide" in o and "  L5-10 Streaming" in o and "  L11-12 Other" in o and "not a heading" not in o, o
        # reading whole files through python is refused
        assert run("run", "python -c \"print(open('docs/guide.md').read())\"").startswith("ERROR: don't read")

    # 13. related: all definitions + per-file usage counts, split into source and tests
    if hasattr(ct, "related"):
        Path(root, "pkg/app.py").write_text("class App:\n    def check(self, d):\n        return self.r.check(d)\n")
        Path(root, "tests/test_app.py").write_text("def test_c(app):\n    app.check(1)\n    app.check(2)\n")
        r = run("related", "check")
        assert "pkg/app.py:2  App.check" in r and "pkg/sse.py:2  ServerSentEvent.check" in r, r
        assert "used in source: pkg/app.py (1)" in r and "used in tests: tests/test_app.py (2)" in r, r

    # 14. find: issue text -> ranked symbols (name matches weigh most), tests listed separately
    if hasattr(ct, "find"):
        import os as _os
        if _os.path.exists("/tmp/ct_index.json"):
            _os.remove("/tmp/ct_index.json")
        r = run("find", "SSE fields must not accept multi-line events; serialize the event check")
        src_part = r.split("tests:")[0]
        assert "ServerSentEvent.check" in src_part or "ServerSentEvent" in src_part, r
        assert "tests:" in r
        _os.remove("/tmp/ct_index.json")

print("ok")
