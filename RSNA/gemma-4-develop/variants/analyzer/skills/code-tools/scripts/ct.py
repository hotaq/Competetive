"""Compact code navigation + escape-tolerant editing for /workspace (stdlib only).

ct.py outline FILE              classes/functions with line ranges
ct.py def NAME                  where NAME (or Class.method) is defined, incl. async
ct.py refs NAME                 lines that use NAME
ct.py show FILE SYMBOL|A-B      exact source of a symbol or line range
ct.py edit FILE OLD NEW [all]   replace OLD with NEW (tolerates \\" style escaping), syntax-checked
ct.py write PATH CONTENT        create/overwrite a file (/tmp allowed)
ct.py run "COMMAND"             run python / pytest / git status|diff in /workspace (no searching)
ct.py grep PATTERN [DIR]        regex search in .py files, max 40 matching lines
"""

import ast
import codecs
import difflib
import os
import re
import shlex
import subprocess
import sys

ROOT = "/workspace"
SKIP = {".git", "__pycache__", "node_modules", "build", "dist", ".venv", "venv", ".tox"}
CAP = 4000  # ~1k tokens per call: the scorer has a 32k context for the whole task


def out(text):
    print(text if len(text) <= CAP else text[:CAP] + f"\n... [cut at {CAP} chars; narrow the query]")


def path_of(p):
    p = p.strip()
    return p if p.startswith("/tmp") else os.path.join(ROOT, p.removeprefix("/workspace/").lstrip("/"))


def rel(p):
    return os.path.relpath(p, ROOT)


def py_files(containing=None):
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.startswith("."))
        for f in sorted(files):
            if f.endswith(".py"):
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


def outline(f):
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


def edit(f, old, new, allow_all=False):
    p = path_of(f)
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
    hits = []
    for p in py_files():
        if p.startswith(base):
            for i, line in enumerate(open(p, errors="ignore"), 1):
                if rx.search(line):
                    hits.append(f"{rel(p)}:{i}: {line.strip()[:120]}")
    more = f"\n... {len(hits) - 40} more; narrow PATTERN or DIR" if len(hits) > 40 else ""
    out("\n".join(hits[:40]) + more if hits else f"no match for {pattern!r} in {where}")


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
        return out("ERROR: `run` only runs python, pytest and git status/diff. To search or read code, ask "
                   "code_analyzer, or use outline/def/refs/show.")
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=240)
    text = (r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else "")).strip()
    if len(text) > CAP:  # keep the end: pytest summaries and tracebacks are at the bottom
        text = f"... [first {len(text) - CAP} chars cut]\n" + text[-CAP:]
    print(f"exit code {r.returncode}\n{text}")


def main(argv):
    cmds = {"outline": (outline, 1), "def": (find_def, 1), "refs": (refs, 1), "show": (show, 2),
            "edit": (edit, 3), "write": (write, 2), "run": (run, 1), "grep": (grep, 1)}
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
