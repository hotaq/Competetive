---
name: exact-output
description: For issues about printed, rendered, serialized or formatted output (text, tables, markup, JSON, headers, repr, docs examples) - hidden tests compare exact strings.
---
- Tests compare the whole string: spaces, trailing newlines, ordering, quoting and escape/ANSI codes all count.
- Reproduce by capturing the real output and printing `repr()` of it; compare it character by character with what the issue says it should be.
- Find where the text is produced (`grep` a fixed fragment of the output), not where it is printed.
- Render with the same options the closest existing test uses (width, color settings, flags); output often depends on them.
- Change only the part the issue names. Don't reformat anything else: neighbouring tests compare other exact strings.
- After the fix, run the whole test file; formatting fixes often break a neighbouring expectation.
