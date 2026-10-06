---
name: add-feature
description: For issues asking for new behaviour (a new parameter, option, function or deprecation), and for issues written as pull-request descriptions rather than bug reports.
---
- The title and description are the spec; skip template text, checklists and links. Every name, default and message they mention is what hidden tests will use.
- New parameter: give it a default that keeps the old behaviour, then thread it through every layer from the public entry point down to where it acts (check the chain with `refs`). Missing one layer is the most common failure.
- Add new keyword arguments at the end; never reorder existing parameters.
- Mirror a sibling: if a similar option already exists, copy its whole path (signature, docstring line, validation, the place it is used).
- Export new public names where their siblings are exported (`__init__.py`, `__all__`).
- Deprecation: keep the old behaviour working and warn with `warnings.warn("...", DeprecationWarning, stacklevel=2)`, worded as in the issue.
- Only change source code: no docs, changelogs or new test files.
