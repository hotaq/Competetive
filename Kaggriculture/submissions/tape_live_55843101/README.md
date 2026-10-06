# Live route corpus — submission 55843101

Recovered 2026-08-29 from public Kaggle replays. **This is a competitor's agent**, owned
by team `boatlee` (id 16641169), not our own work. Our team is `zero waste` (id 16621481).

## Fidelity: PARTIAL — this is not a faithful reconstruction

Unlike `tape_live_55861364`, this agent is **not** a single fixed program. Five sampled
replays show at least two distinct route families:

| Episode | Seat | Seed | Opponent | Final money |
|---|---:|---:|---|---:|
| 102409057 | 0 | 1305847007 | Chika Komari | 89,142 |
| 102458235 | 0 | 1529582062 | JeremiahMannings | 129,722 |
| 102516290 | 1 | 140811240 | Efnutrrionpy | 95,855 |
| 102536309 | 1 | 19840569 | Xiaowenhao404 | 106,069 |
| 102556219 | 0 | 504753283 | Masashi Onda | 82,981 |

Episode 102409057 diverges from every other sample at **step 2** and shares only 0.3% of
its actions. The remaining four agree with each other on 89.6-96.0% of steps, diverging at
steps 89, 151, and 218. Only 2 of 720 steps are common to all five episodes, and divergence
does not track seat.

This is the signature of a state-conditioned multi-route policy, consistent with the public
notebooks `boatlee/v21-r1-public-state-route-portfolio` and
`boatlee/v20-adaptive-r1-multi-route-agent`.

**Therefore the route selection logic cannot be recovered from replays.** These files
preserve five observed executions, not the agent. Since this is a competitor's notebook,
there is no way to obtain the source, and none is needed: it is a benchmark, not our work.

## Contents

- `routes/route_ep<id>.json` — one captured 720-action execution each
- `routes_index.json` — provenance and SHA-256 per route
