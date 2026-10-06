"""Compact code navigation + escape-tolerant editing for /workspace (stdlib only).

ct.py find "WORDS"              rank functions/classes by relevance to free text (e.g. the issue): start here
ct.py outline FILE              classes/functions (.py) or headings (.md) with line ranges
ct.py def NAME                  where NAME (or Class.method) is defined, incl. async
ct.py refs NAME                 lines that use NAME
ct.py related NAME              one-call map: every definition of NAME, and the files that use it (source/tests)
ct.py show FILE SYMBOL|A-B      exact source of a symbol or line range
ct.py edit FILE OLD NEW [all]   replace OLD with NEW (tolerates \\" style escaping), syntax-checked
ct.py write PATH CONTENT        create/overwrite a file (/tmp allowed)
ct.py run "COMMAND"             run python / pytest / git status|diff in /workspace (no searching)
ct.py grep PATTERN [DIR]        regex search in source, docs and config files, max 40 matching lines
"""

import ast
import codecs
import collections
import difflib
import json
import math
import os
import re
import shlex
import subprocess
import sys

ROOT = "/workspace"
SKIP = {".git", "__pycache__", "node_modules", "build", "dist", ".venv", "venv", ".tox"}
CAP = 3000  # <1k tokens per call: the scorer has a 32k context (minus 4k output) for the whole task
TEXT = {".py", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".html", ".js", ".css"}


def out(text):
    print(text if len(text) <= CAP else text[:CAP] + f"\n... [cut at {CAP} chars; narrow the query]")


def path_of(p):
    p = p.strip()
    return p if p.startswith("/tmp") else os.path.join(ROOT, p.removeprefix("/workspace/").lstrip("/"))


def rel(p):
    return os.path.relpath(p, ROOT)


def py_files(containing=None, exts=(".py",)):
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.startswith("."))
        for f in sorted(files):
            if f.endswith(tuple(exts)) and not f.startswith("."):  # skips ADK's .adk_exec_*.py (holds this skill + args)
                p = os.path.join(d, f)
                if containing is None or containing in open(p, errors="ignore").read():
                    yield p


def symbols(tree):
    """(qualname, node) for every class/function, nested ones included."""
    stack = [("", n) for n in tree.body]
    while stack:
        prefix, n = stack.pop(0)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            q = prefix + n.name
            yield q, n
            stack += [(q + ".", c) for c in n.body]


def start(n):
    return min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno])


def head(n):
    if isinstance(n, ast.ClassDef):
        return f"class {n.name}({', '.join(ast.unparse(b) for b in n.bases)})"
    kw = "async def" if isinstance(n, ast.AsyncFunctionDef) else "def"
    sig = ast.unparse(n.args)
    return f"{kw} {n.name}({sig if len(sig) < 90 else sig[:87] + '...'})"


def parse(p):
    try:
        return ast.parse(open(p, errors="ignore").read())
    except SyntaxError:
        return None


def matches(q, name):
    return q == name or q.endswith("." + name)


def md_outline(p):
    """Markdown headings with the line range each section spans (fenced code is skipped)."""
    heads, fence = [], False
    lines = open(p, errors="ignore").read().splitlines()
    for i, line in enumerate(lines, 1):
        fence ^= line.lstrip().startswith("```")
        m = None if fence else re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            heads.append((len(m.group(1)), i, m.group(2).strip()))
    rows = []
    for k, (level, i, title) in enumerate(heads):
        end = next((j - 1 for lv, j, _ in heads[k + 1:] if lv <= level), len(lines))
        rows.append(f"{'  ' * (level - 1)}L{i}-{end} {title[:100]}")
    return out("\n".join(rows) or "no headings")


def outline(f):
    if f.endswith((".md", ".markdown")):
        return md_outline(path_of(f))
    tree = parse(path_of(f))
    if tree is None:
        return out(f"{f}: cannot parse")
    lines = [f"{'  ' * q.count('.')}L{start(n)}-{n.end_lineno} {head(n)}" for q, n in sorted(symbols(tree), key=lambda s: start(s[1]))]
    out("\n".join(lines) or "no classes or functions")


def find_def(name):
    hits = []
    for p in py_files(name.split(".")[-1]):
        tree = parse(p)
        for q, n in symbols(tree) if tree else []:
            if matches(q, name):
                hits.append(f"{rel(p)}:{n.lineno}-{n.end_lineno}  {q}  {head(n)}")
        for n in tree.body if tree else []:  # module-level constants / aliases
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                hits.append(f"{rel(p)}:{n.lineno}  {ast.unparse(n)[:120]}")
    out("\n".join(hits[:40]) or f"no definition of {name}")


def refs(name):
    short, hits = name.split(".")[-1], []
    for p in py_files(short):
        tree = parse(p)
        src = open(p, errors="ignore").read().splitlines()
        seen = set()
        for n in ast.walk(tree) if tree else []:
            hit = (isinstance(n, ast.Name) and n.id == short) or (isinstance(n, ast.Attribute) and n.attr == short) \
                or (isinstance(n, ast.alias) and n.name.split(".")[-1] == short)
            if hit and n.lineno not in seen:
                seen.add(n.lineno)
                hits.append(f"{rel(p)}:{n.lineno}: {src[n.lineno - 1].strip()[:110]}")
    extra = f"\n... {len(hits) - 50} more" if len(hits) > 50 else ""
    out("\n".join(sorted(hits)[:50]) + extra if hits else f"no references to {short}")


STOP = set("the and for that this with from not are was but have has you your its can will should would when what "
           "into than then them they which there their been being does did also only just some such self none true "
           "false return def class import pass else elif while yield async await raise try except finally lambda".split())


def terms(text):
    """Identifier-aware words: `serializeResponse` / `serialize_response` -> serialize, response."""
    out_ = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w.replace("_", " ")):
            part = part.lower()
            if len(part) > 2 and part not in STOP:
                out_.append(part)
    return out_


def index():
    """Every class/function as (file, start, end, qualname, header, name terms, body terms); cached in /tmp."""
    cache = "/tmp/ct_index.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    rows = []
    for p in py_files():
        tree = parse(p)
        src = open(p, errors="ignore").read().splitlines()
        for q, n in symbols(tree) if tree else []:
            body = "\n".join(src[n.lineno - 1:min(n.end_lineno, n.lineno + 40)])
            rows.append([rel(p), start(n), n.end_lineno, q, head(n), terms(q + " " + rel(p)[:-3]), terms(body)])
    json.dump(rows, open(cache, "w"))
    return rows


def find(text):
    q = collections.Counter(terms(text))
    for span in re.findall(r"`([^`\n]{1,80})`", text):  # names the issue puts in backticks matter most
        q.update({t: 2 for t in terms(span)})
    rows = index()
    df = collections.Counter(t for r in rows for t in set(r[5] + r[6]))
    idf = {t: math.log(len(rows) / (1 + df[t])) for t in q}
    scored = []
    for f, a, b, qual, hd, name_t, body_t in rows:
        tf = collections.Counter(body_t)
        s = sum(q[t] * idf[t] * (3 * name_t.count(t) + math.log1p(tf[t])) for t in q if t in tf or t in name_t)
        if s > 0:
            scored.append((s, f, a, b, qual, hd))
    scored.sort(reverse=True)
    src = [x for x in scored if not is_test_file(os.path.join(ROOT, x[1]))][:10]
    tst = [x for x in scored if is_test_file(os.path.join(ROOT, x[1]))][:4]
    fmt = lambda x: f"  {x[1]}:{x[2]}-{x[3]}  {x[4]}  {x[5][:90]}"
    out("source:\n" + ("\n".join(map(fmt, src)) or "  none") + "\ntests:\n" + ("\n".join(map(fmt, tst)) or "  none"))


def related(name):
    """Map a symbol across the repo so multi-file changes start from a file list, not from reading files."""
    short, defs, uses = name.split(".")[-1], [], collections.Counter()
    for p in py_files(short):
        tree = parse(p)
        if tree is None:
            continue
        defs += [f"  {rel(p)}:{n.lineno}  {q}" for q, n in symbols(tree) if matches(q, name)]
        lines = {n.lineno for n in ast.walk(tree)
                 if (isinstance(n, ast.Name) and n.id == short) or (isinstance(n, ast.Attribute) and n.attr == short)
                 or (isinstance(n, ast.alias) and n.name.split(".")[-1] == short)}
        if lines:
            uses[rel(p)] = len(lines)
    test = {f: c for f, c in uses.items() if is_test_file(os.path.join(ROOT, f))}
    src = {f: c for f, c in uses.items() if f not in test}
    top = lambda d, k: ", ".join(f"{f} ({c})" for f, c in sorted(d.items(), key=lambda x: -x[1])[:k]) + (
        f", ... {len(d) - k} more files" if len(d) > k else "")
    out("defined:\n" + ("\n".join(defs[:12]) or "  (no definition found)")
        + f"\nused in source: {top(src, 12) or 'none'}\nused in tests: {top(test, 8) or 'none'}")


def show(f, what):
    p = path_of(f)
    lines = open(p, errors="ignore").read().splitlines()
    if what.replace("-", "").replace(":", "").isdigit():
        a, _, b = what.replace(":", "-").partition("-")
        a, b = int(a), int(b or a)
    else:
        tree = parse(p)
        found = [(start(n), n.end_lineno) for q, n in symbols(tree) if matches(q, what)] if tree else []
        if not found:
            return out(f"{what} not found in {f}; try: ct.py outline {f}")
        a, b = found[0]
    out(f"# {rel(p)} lines {a}-{b} (copy text from here for edits; no line numbers inside)\n" + "\n".join(lines[a - 1:b]))


# The text as sent, then the usual JSON-escaping slips: \\" for ", halved or doubled backslashes.
TRANSFORMS = [
    lambda s: s,
    lambda s: s.replace('\\"', '"').replace("\\'", "'"),
    lambda s: s.replace("\\\\", "\\"),
    lambda s: s.replace("\\", "\\\\"),
    lambda s: codecs.escape_decode(s.encode())[0].decode(),
]


def variants(s):
    """(transform, transformed text) pairs; the same transform is later applied to NEW."""
    for t in TRANSFORMS:
        try:
            yield t, t(s)
        except Exception:
            pass


def syntax_error(p, text):
    if not p.endswith(".py"):
        return None
    try:
        compile(text, p, "exec")
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"


def is_test_file(p):
    """Test files are reset before grading; an edit to one makes the hidden tests fail to apply."""
    if p.startswith("/tmp"):
        return False
    name, parts = os.path.basename(p), rel(p).split(os.sep)
    return (name.startswith("test_") or name.endswith("_test.py") or name in {"conftest.py", "pytest.ini"}
            or "tests" in parts[:-1] or "test" in parts[:-1])


TEST_REFUSED = ("ERROR: refused to change a test file. Test files are reset before grading, and a changed one makes the "
                "hidden tests fail to apply, so the whole task fails. If an existing test asserts the old behaviour the "
                "issue asks to change, leave it: the hidden tests replace it. Change source code only.")


def edit(f, old, new, allow_all=False):
    p = path_of(f)
    if is_test_file(p):
        return out(TEST_REFUSED)
    src = open(p, errors="ignore").read()
    for t, v in variants(old):
        if v and v in src:
            count = src.count(v)
            if count > 1 and not allow_all:
                at = [src[:i].count("\n") + 1 for i in _positions(src, v)]
                return out(f"ERROR: OLD occurs {count} times (lines {at}); add surrounding lines to make it unique, or pass 'all'")
            new_v = t(new)
            result = src.replace(v, new_v)
            err = syntax_error(p, result)
            if err:
                return out(f"ERROR: not saved, the edit would make {rel(p)} invalid Python ({err}). Check quotes/indentation in NEW.")
            open(p, "w").write(result)
            diff = difflib.unified_diff(src.splitlines(), result.splitlines(), rel(p), rel(p), lineterm="", n=1)
            return out(f"OK: {count} replacement(s) in {rel(p)}" + ("" if v == old else " (escaping in OLD/NEW was normalized)")
                       + "\n" + "\n".join(list(diff)[2:]))
    lines, n = src.splitlines(), max(1, old.count("\n") + 1)
    best = max(range(max(1, len(lines) - n + 1)), key=lambda i: difflib.SequenceMatcher(None, "\n".join(lines[i:i + n]), old).ratio())
    out(f"ERROR: OLD not found in {rel(p)}. Closest text is lines {best + 1}-{best + n}; copy OLD exactly from it:\n"
        + "\n".join(lines[best:best + n]))


def _positions(s, sub):
    i = s.find(sub)
    while i >= 0:
        yield i
        i = s.find(sub, i + 1)


def write(f, content):
    p = path_of(f)
    if is_test_file(p):
        return out(TEST_REFUSED)
    for _, v in variants(content):
        if not syntax_error(p, v):
            content = v
            break
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(content)
    note = "" if p.startswith("/tmp") else " (inside /workspace: this file WILL be in your patch)"
    out(f"OK: wrote {p} ({len(content)} chars){note}" + (f"\nWARNING: {syntax_error(p, content)}" if syntax_error(p, content) else ""))


def grep(pattern, where="."):
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))
    base = os.path.normpath(os.path.join(ROOT, where.removeprefix("/workspace/").lstrip("/")))
    groups = {"source": [], "tests": [], "docs/other": []}
    for p in py_files(exts=TEXT):
        if p.startswith(base):
            kind = "tests" if is_test_file(p) else "source" if p.endswith(".py") and "docs" not in rel(p).split(os.sep)[:1] else "docs/other"
            for i, line in enumerate(open(p, errors="ignore"), 1):
                if rx.search(line):
                    groups[kind].append(f"{rel(p)}:{i}: {line.strip()[:120]}")
    # Library code before example dirs (docs_src/, examples/): same hits, better order.
    groups["source"].sort(key=lambda h: h.split("/")[0] in {"docs_src", "examples", "example", "docs", "tutorial"})
    # Code first: big docs trees otherwise fill the cap and hide the source hits.
    show_docs = where != "." or not (groups["source"] or groups["tests"])
    lines, room = [], 40
    for kind in ("source", "tests", "docs/other"):
        hits = groups[kind]
        if not hits:
            continue
        if kind == "docs/other" and not show_docs:
            lines.append(f"+ {len(hits)} matches in docs/other files (pass a DIR such as `docs` to see them)")
            continue
        lines += [f"[{kind}]"] + hits[:room] + ([f"... {len(hits) - room} more {kind} matches; narrow PATTERN or DIR"] if len(hits) > room else [])
        room = max(room - len(hits), 5)
    out("\n".join(lines) if lines else f"no match for {pattern!r} in {where}")


RUNNABLE = {"python", "python3", "pytest"}
GIT_OK = {"status", "diff"}


def run(cmd):
    """Run/verify only. Searching belongs to code_analyzer, whose context is discarded afterwards."""
    try:
        words = shlex.split(cmd)
    except ValueError as e:
        return out(f"ERROR: {e}")
    first = os.path.basename(words[0]) if words else ""
    ok = first in RUNNABLE or (first == "git" and len(words) > 1 and words[1] in GIT_OK)
    if not ok or any(w in {"grep", "egrep", "rg", "find", "cat", "ls", "head", "tail", "sed", "awk"} for w in words):
        return out("ERROR: `run` only runs python, pytest and git status/diff. To search or read files use "
                   "grep, outline, def, refs and show.")
    if re.search(r"open\(.*\)\s*\.read|listdir|os\.walk", cmd):  # printing whole files/trees floods the context
        return out("ERROR: don't read or list files through python. Use `outline FILE` for its structure, then "
                   "`show FILE A-B` for the lines you need, and `grep` to find things.")
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=240)
    text = (r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else "")).strip()
    if len(text) > CAP:  # keep the end: pytest summaries and tracebacks are at the bottom
        text = f"... [first {len(text) - CAP} chars cut]\n" + text[-CAP:]
    print(f"exit code {r.returncode}\n{text}")


def main(argv):
    cmds = {"outline": (outline, 1), "def": (find_def, 1), "refs": (refs, 1), "show": (show, 2),
            "edit": (edit, 3), "write": (write, 2), "run": (run, 1), "grep": (grep, 1), "related": (related, 1), "find": (find, 1)}
    while argv and argv[0] in {"run_skill_script", "code-tools", "scripts/ct.py", "ct.py"}:
        argv = argv[1:]  # the tool's own parameters sometimes get pasted into args
    if not argv or argv[0] not in cmds or len(argv) - 1 < cmds[argv[0]][1]:
        return out(__doc__)
    fn, n = cmds[argv[0]]
    args = argv[1:1 + n]
    if argv[0] == "edit" and len(argv) > 4:
        args.append(argv[4] == "all")
    if argv[0] == "grep" and len(argv) > 2:
        args.append(argv[2])
    try:
        fn(*args)
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        out(f"ERROR: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
