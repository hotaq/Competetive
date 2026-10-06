# Newest public agents vs N3 / SB682 (pre-registered 2026-09-26, before any game)

Live at 05:30 UTC: N3 (56521684) 2370.6, SB682 (56541172) 2280.5, H54c 2308.4.
Code scanned statically: no network or subprocess calls; `open(path)` only reads an
optional, unset library path.

Candidates (artifacts/public_notebooks_20260926/outputs, md5[:8]):
P1 harvest-ledger 49196f58 · P2 multi-route v70 252038e5 · P3 top-2-master-engine-v4 851d9c90 ·
P4 kaggriculture-hack v38 4593a884 · P5 shepherds-ledger 2e46a53c · P6 evgendvorkin v31 9e1b47a0.

Stage 1: each P vs N3 and vs SB682, seeds 26100001–26100020 both seats (40 each), plus the
N3-live panel (136, `analysis/n3_live_20260925/replays`; N3 0.5551, SB682 0.6397) and the
C-live panel (68; N3 0.8088). Eligible: N3-live point >= 0.6397 (at least SB682's).
W = the eligible candidate with the highest h2h vs N3; it must be >= 0.55.

Stage 2 (W):
| gate | criterion |
|---|---|
| M1 | h2h vs N3, seeds 26200001–26200100 both seats: point >= 0.60, CI95 lower > 0.50 |
| M2 | h2h vs SB682 on the same seeds: point >= 0.50 |
| M3 | N3-live point >= 0.6397 and C-live >= 0.8088 − 0.02 (from stage 1) |
| M4 | 0 exceptions |
| G3 | official env from W's own tarball, both seats vs N3, money equals play.py |

Pass → propose. Fail → keep current submissions.
