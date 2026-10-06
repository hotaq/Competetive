# E019 — V104 plus EfficientNet-B3 targetwise ranks

E019 retains exact E014/V104 and adds the public five-fold EfficientNet-B3
deployment as a genuinely different convolutional family.

The attached public audit reports Gold58 macro AUC `0.84348 -> 0.85481` for its
nested targetwise recipe. E019 uses those fixed target weights, disables the
older legacy blend, and also writes a conservative global 10% diagnostic.

Submission artifact:

- `submission_target_plus_b3_targetwise.csv`

The notebook fails closed if B3 inference is unavailable or times out; it will
not silently submit the unchanged V104 parent.
