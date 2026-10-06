# E018 — V104 Rad guard sweep

This experiment keeps the scored E014/V104 `submission.csv` byte-path intact
and emits two controlled variants from the same already-computed predictions:

- `submission_fracture_rad.csv`: Baker's remains guarded; Fracture receives the
  E10 RadImageNet vote.
- `submission_all_rad.csv`: neither Baker's nor Fracture is guarded at E10.

The Fracture hypothesis is grounded in Dariush Afshar's related full-population
OOF audit and public-LB A/B (`0.920 -> 0.921`). The all-target variant tests the
less-certain Baker's extension separately. Both outputs pass schema, UID,
finite/range, declared-recipe, and SHA-256 checks.
