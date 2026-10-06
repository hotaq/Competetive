# V86-A Family-Selection Freeze Supplement

Frozen before any V86 candidate game, reward, margin, or economic outcome was generated. The selector reads preexisting snapshot rank/team identifiers and static public tape actions; it does not read any new V86 rewards or outcomes, nor any V67/B/C rewards or margins. This supplement makes the Top20 family-selection rule in the original pre-registration fully executable and reproducible; it does not change the selected families, seeds, actions, model, or gates.

- Selector: `select_families.py`
- Selector SHA-256: `79cb474b7a266207d557c69dd2d30a74bf3ac9199fd248dc72d26acf7c0c226f`
- Selection artifact: `family_selection.json`
- Selection artifact SHA-256: `9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38`
- 80-tape source manifest SHA-256: `d3b17f42fa2839e0e86a25f29b96bb5661e739d4abd82154c3b3818a6a6c2947`
- Top20 rank manifest SHA-256: `1253ef13a3bea8bfd8e741185d27c178b296e3aafde08832972dd5dd16934c57`

Two independent root reruns and the frozen artifact were byte-identical at SHA-256 `9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38`.

Selected submissions remain exactly `55714246`, `55734995`, `55774239`, and `55742738`; ordered backups remain `55616096`, then `55617399`. Selection read preexisting snapshot ranks, team identifiers, and static public tape actions. It ran no games and read no new V86 rewards or outcomes, no V67/B/C rewards or margins, and no sealed holdout data.
