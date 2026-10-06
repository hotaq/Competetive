# Closed-loop arena (pre-registered 2026-09-24, before any arena game)

User goal: 80% win rate vs top-10. The recorded-top-10 panels cannot measure
that: C scores 0.835 on T24 but the yannik tape (live 1850) scores 0.772, and
the opponents' money falls 30% (106.7k live → 74.7k) because their recorded
actions cannot react. The only reactive opponents we can run are public
closed-loop agents. aurax7 (team 摆烂小分队, rank 26, 2876.9) is the only
top-30 member with public code; the lineage C belongs to is derived from it.

## Field (frozen)

C (submitted 56513871), A, D, E, F, G (see PREREGISTRATION.md; B plays
identically to C, H is weakest; both dropped), X7 = aurax7
shop-router-reactive-v7, X5 = aurax7 shop-router-reactive-v5, Y = yannik tape.

## Protocol

Full round-robin, closed-loop vs closed-loop in `play.py` (`vs|…`), seeds
25000001–25000050, both seats (100 games per pair). Fidelity check first: one
C-vs-X7 game in the official env from files must equal play.py money exactly.

## Reading (frozen)

- Rank by pooled point over all opponents.
- Challenger rule: an agent X replaces C as the Slot-B candidate only if
  h2h(X vs C) >= 0.55 with a seed-cluster bootstrap CI95 lower bound > 0.50 AND
  X's pooled point > C's. It must then pass P1-P3/G3 on the live panels before
  being proposed. Otherwise C stays.
- Per-agent point vs X7/X5 is reported as the best available proxy for
  "vs near-top-10 reactive play".
