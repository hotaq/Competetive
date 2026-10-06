# Generalization check for N3 (descriptive; pre-registered 2026-09-24 before any game)

Question (user): does our agent generalize to a new meta? N3 (56521684) was selected
on h2h vs H54c plus the three live panels. This run plays N3, and H54c for comparison,
against opponent families that played no part in selecting N3, on fresh seeds
25700001–25700030, both seats (60 games per opponent).

Families:
- F1 newest public (published 2026-09-24): N1, N2, N4, N5, N6, N7, N9 (N8 = N5)
- F2 older public (Sept 17-23): A, D, E, F, G, X7, X5
- F3 evolved clones of our own lineage: C, H54c/N3 (the other arm), round-1 V (race48+ca−7),
  race38, race44, H72c, R48c11
- F4 top-10 recorded strategies (open-loop tapes, best game per team from the
  Sept-24 pool): DSM, DECEM, Vadim, M&M&P&Q, THIRD FARM, Majkel, 吃白饭的大肥鱼,
  Kaggledew, Unknown Mother-Goose, Fourth Quadrant

Reported per family: mean point, worst opponent, count of opponents below 0.50.
Flag (frozen): N3 "does not generalize" if any family mean < 0.60 or any single
opponent < 0.30. No selection or submission decision is attached.
