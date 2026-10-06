---
name: write-failing-test
description: How to write /tmp/test_issue.py so it fails for the issue's reason - exact exception messages, captured output, warnings, async code, and working without the repo's fixtures.
---
- Start from a copy, never from scratch: `show` the top of the closest test file (imports and module-level setup) and the closest test function, and copy them into /tmp/test_issue.py. Then change only the input and the assertion.
- Framework code (web apps, CLIs, ORMs, plugins) is tested through the setup the repo's tests already use (the app/route/test client, runner or session they build). Copy that setup exactly; an invented setup usually tests something else.
- Fixtures from the repo's `conftest.py` are NOT available in /tmp: copy the fixture's body into the test instead.
- Exceptions: `with pytest.raises(ValueError, match=re.escape("exact message")):` (match is a regex searched in `str(exc)`, so escape it).
- Output: capture it into a string and compare the whole thing (`assert out == "..."`). When it fails, print `repr(out)` once to see spaces, newlines and escape codes.
- Warnings: `with pytest.warns(DeprecationWarning, match="..."):`.
- Async code: call `asyncio.run(main())` inside a normal test function; no plugin needed.
- Several inputs: `@pytest.mark.parametrize("value", [...])` with every case the issue names.
- Before the fix it must fail with the problem the issue describes. An ImportError, NameError or TypeError raised by the test itself means the test is wrong: fix the test.
- If it passes before the fix, that path already works: the bug is on another way into the same feature. Features usually have several (a single parameter vs a model/class/config object that groups them, sync vs async, one value vs a list, a function vs a method). List them, then test the next one. The closest existing test file's name often says which path the issue is about.
- After two passing attempts, stop writing tests: `grep` where the feature's option is read in the source and check each code path that reads it.
- Keep one test. Once /tmp/test_issue.py fails for the issue's reason, it is your only check: don't write other test files, don't mock internal functions, don't test third-party libraries.
- Trust the environment: it is offline and correctly installed. If a library seems broken, your usage is wrong (for example a real network call).
- A fix that doesn't change the test result is on the wrong line: `show` the rest of the function handling the value (especially loops after your edit) and trace where the bad value still gets through.
