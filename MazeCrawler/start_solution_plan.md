# Maze Crawler Start Solution Plan

## Goal

Create a simple but measurable first agent that beats the starter policy by improving exploration, avoiding friendly fire, and spending build energy only when useful.

## Baseline

The Kaggle starter builds workers when possible, moves north, and removes north walls with workers. It does not remember intent, coordinate units, seek crystals/nodes, avoid friendly collisions, or infer enemy movement from symmetry.

## Candidate Lane 1: Safer Starter

- Keep the factory moving north when passable.
- Use `JUMP_NORTH` only when the landing row is safely inside the visible board.
- Avoid spawning when the cell north of the factory is occupied by a friendly unit.
- Workers remove north walls only when the target is not fixed and movement north is otherwise blocked.
- Non-factory units prefer north, then visible crystals, then unexplored passable cells.

Status: implemented in `main.py` as v1b safe starter. It adds collision-aware reservations, safer spawn checks, emergency north jumps, conservative worker wall removal, deterministic lateral fallback, simple crystal/node scoring, and fallback observation bounds for replay/runtime shapes that omit `southBound`/`northBound`.

## Candidate Lane 2: Scout-First Mapper

- Build scouts early because they are cheap, fast, and high-vision.
- Send scouts to different lateral lanes to reveal walls, crystals, and mining nodes.
- Keep factory route conservative while scouts reveal the next corridor.
- Add collision reservations so same-type friendly units do not move into the same cell.

## Candidate Lane 3: Mine Economy

- Build a miner only after a mining node is visible or recently inferred reachable.
- Move miner to node and `TRANSFORM` when energy is enough to pay transform cost.
- Use nearby robots to collect mine energy and transfer back toward the factory when useful.

## Measurement

- Run each candidate against `random` over fixed seeds first.
- Then run candidate-vs-starter self-play with swapped player order.
- Track win rate, final reward, factory survival step, total energy, unit count, and average collisions.

## Submission Rule

Submit only a `main.py` at root unless we add helper files, in which case package `main.py` plus helpers into `submission.tar.gz`.
