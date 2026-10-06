# V86-A Contract Freeze Receipt V2

This receipt supersedes `FREEZE_RECEIPT.md` after a pre-outcome wording correction. No V86 game or V67/B/C candidate economic outcome had been generated when V2 was frozen. The selected families, seeds, actions, model, CV, and gates did not change.

- Machine protocol `protocol.json`: `007e36ed80bc2f08cb1cd2495db00ce4bd5132804f761550bae0b72326ef3b30`
- Corrected pre-registration `PRE_REGISTRATION.md`: `875791c2811ffcb1f464b8daeb8959c0626d1341cbed5a8763ccd16cbe4aea6c`
- Family selector `select_families.py`: `79cb474b7a266207d557c69dd2d30a74bf3ac9199fd248dc72d26acf7c0c226f`
- Family selection `family_selection.json`: `9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38`
- Family-selection supplement `FREEZE_SUPPLEMENT_FAMILY_SELECTION.md`: `16e1826923234a049cff74631daba878723050f4878174926a45c39502724060`
- Frozen V67: `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`

The corrected provenance explicitly records that the selector read pre-existing public snapshot rank and public tape actions. It did not read any new V86 candidate outcomes or any V67/B/C rewards or margins, and it did not run a game or open the sealed holdout. Root reproduced the corrected family-selection artifact twice byte-for-byte at SHA-256 `9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38`.
