---
name: errors-and-validation
description: For issues about input that should raise (or stop raising) an error, about error messages or exception types, or about validation rules.
---
- Hidden tests usually use `pytest.raises(<Type>, match=<message>)`: both the exception type and the message words matter.
- Exception type: follow the issue; otherwise use the type that sibling checks in the same module raise (`grep "raise " <file>`).
- Message: state the rule in the issue's own words (title and description), in the style of the module's existing messages (quoting, field names).
- Validate at the one place every input passes through (constructor, validator, setter, parser), not in each caller.
- Cover every invalid form the issue names, and keep valid inputs (None, defaults) working.
- If something should stop raising, narrow the condition of the check that raises; don't wrap callers in try/except.
