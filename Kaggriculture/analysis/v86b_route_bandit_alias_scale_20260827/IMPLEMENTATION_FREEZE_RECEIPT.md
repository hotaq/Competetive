# V86-B Implementation Freeze Receipt

Frozen before any V86-B game or economic outcome was generated.

- Collection runner `run_experiment.py`: `f4cb36b06c2fc59b34fa12995172d725d28c40eeed6bfcf460a2f342d7d59add`
- Fixed evaluator `evaluate_cv.py`: `713c3554ddf28670d68eccd120e121098b12a4964456ef4baafeee595145819e`
- Machine protocol `protocol.json`: `511152132c5fa6f245ee8eabb10f29d60007ddd79a7c3b3d523ba7dfa9ee5731`
- Pre-registration `PRE_REGISTRATION.md`: `a1e442116f1c19aacd50647b0df4d9d3ffb08cac11b8e408ba16498601076e3b`
- Contract receipt `CONTRACT_FREEZE_RECEIPT.md`: `d0c1462ac6234590df6416f5eb2355e2f4585e535ebefa00ac246f8a655bafe3`
- Frozen V67: `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- V86-A stop evidence only `STOP_REPORT.md`: `6b01909a87d1b7dba13379a000a122f559848d52a14a419993ddfa3a7f30feb9`

Pre-outcome static validation passed: runner protocol verification matched 28/28 pins; full-joint prefix, route lifecycle, alias truth-table and executed-action support logic passed adversarial source audit; both CLIs expose only the frozen B interfaces; Ruff check and format pass; and evaluator numerical parity/self-test passes. The V86-A dataset/results were not opened or parsed by V86-B implementation. No V86-B game, reward, margin, checkpoint, runtime agent, sealed-holdout read, submission, or upload occurred before this receipt.
