# V86-B Contract Freeze Receipt V4

Status: **FROZEN BEFORE ANY V86-B GAME OR ECONOMIC OUTCOME** on 2026-08-27.

This final joint-prefix-strengthened receipt supersedes the earlier V86-B drafts after pre-outcome critic review. No V86-B game was run, no V86-B label or margin existed, and no V86-A economic dataset, result, label, margin, reward, score, oracle, rescue, selector, or CV artifact was opened, parsed, read, or used while preparing this contract.

## Contract artifacts

- `protocol.json`: SHA-256 `511152132c5fa6f245ee8eabb10f29d60007ddd79a7c3b3d523ba7dfa9ee5731`
- `PRE_REGISTRATION.md`: SHA-256 `a1e442116f1c19aacd50647b0df4d9d3ffb08cac11b8e408ba16498601076e3b`

`CONTRACT_FREEZE_RECEIPT.md` is intentionally not self-hashed inside itself. Its delivered hash is reported externally after creation.

## V86-A exclusion evidence

- The only V86-A outcome-related artifact permitted to be opened or parsed is the mechanical stop report `analysis/v86_route_bandit_family_scale_20260827/STOP_REPORT.md`, SHA-256 `6b01909a87d1b7dba13379a000a122f559848d52a14a419993ddfa3a7f30feb9`.
- Its sole permitted use is to establish that V86-A stopped before CV on the original non-null first-difference predicate.
- V86-A `dataset.json`, `dataset_smoke.json`, and every other V86-A result or economic artifact must not be opened, read, hashed, imported, parsed, copied, aggregated, inspected, or used by V86-B.

## Frozen current source manifest

| Role | Path | SHA-256 |
|---|---|---|
| V86-A protocol reference | `analysis/v86_route_bandit_family_scale_20260827/protocol.json` | `007e36ed80bc2f08cb1cd2495db00ce4bd5132804f761550bae0b72326ef3b30` |
| V86-A pre-registration reference | `analysis/v86_route_bandit_family_scale_20260827/PRE_REGISTRATION.md` | `875791c2811ffcb1f464b8daeb8959c0626d1341cbed5a8763ccd16cbe4aea6c` |
| V86-A family selection | `analysis/v86_route_bandit_family_scale_20260827/family_selection.json` | `9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38` |
| V86-A family selector | `analysis/v86_route_bandit_family_scale_20260827/select_families.py` | `79cb474b7a266207d557c69dd2d30a74bf3ac9199fd248dc72d26acf7c0c226f` |
| Collection reference | `analysis/v86_route_bandit_family_scale_20260827/run_experiment.py` | `30faa31202d938d781413843bf572fa77ade04e4bf45081bf0b884230ca09412` |
| CV reference | `analysis/v86_route_bandit_family_scale_20260827/evaluate_cv.py` | `4a99d63638e862572f4afb5ba1296ba5a218d4b09e85b663b2f7522ce6a5ccf4` |
| V85 collector reference | `analysis/v85_route_bandit_cv_20260827/collect_features.py` | `a5d69bd1c447f7b0f5a5eebf0020a71163778e66dc5bedd9b999771e3d1749ef` |
| V84 route executor | `analysis/v84_route_option_onpolicy_oracle_20260827/run_oracle.py` | `9fc45c32923aeba7891a04a2d4e09e4f6b159266b1c7a81ecb42761deb5081a6` |
| Selector encoder | `src/kaggriculture_rl/learning/tape_selector.py` | `71172b8a853d2e63798e20c6e2cd7b64763d79d4c11e9ba01822669a0382c31f` |
| Engine helper | `scripts/evaluate_file_agent_vs_replay.py` | `03fcb73cd657e09b4c74498e71ad093813d16c8dad669aeb78f4013b2ed9966f` |
| Rust Python package | `bindings/python/kaggriculture_rs/__init__.py` | `672f7aa733ccdf6146bb001b8f465ffbb9a94d5e1bfed3f4041ff0e8a5688adb` |
| Rust native extension | `bindings/python/kaggriculture_rs/_native.abi3.so` | `315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd` |
| Frozen V67 | `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py` | `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b` |
| B120 route tape | `artifacts/top20_live_tapes80_20260826/tape_99600846_s0.py` | `3fdf8cc7634eb9f60fcdecaefdd62ac95d3f04509d0c2bb4f82bebb1c763b627` |
| C120 route tape | `artifacts/top20_live_tapes80_20260826/tape_99596430_s1.py` | `6510a81b1114e4cbdf899afb194d3bd015abcf8cd010579a8de4a0fe93d9226f` |
| `v16_route_family` | `submissions/meta_v22_v16_land3_terminal_rule_standalone/main.py` | `5d5c615a296d984402f886b5f998a54e211dd6138e7818793733069d0d1476ba` |
| `ryo_selector_family` | `submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py` | `670c85ca7c81d150fbebb4083eaec37f667c3c7305dfdc55a41feed3919e4d02` |
| `route_99612231_family` | `submissions/meta_v44_route99612231_orphan_standalone/main.py` | `cd2eda3600b4d4849f40a74d25f350239528541e52be0aecf85215fe7aa9afb2` |
| `robust_cluster_family` | `submissions/robust_cluster_tape_v10_standalone/main.py` | `27cb59ba30d42464985db5fa58f858f79c6ddae95e1eeff2bd4b4275523805f6` |
| `top20_crop_dusta_55714246` | `artifacts/top20_live_tapes80_20260826/tape_99594187_s1.py` | `5caefeb46cec9f03f5ca80c1ba32bc57e294370685ae06d12ab7be202590a75a` |
| `top20_tetsuya_55734995` | `artifacts/top20_live_tapes80_20260826/tape_99591726_s0.py` | `f5ac53cd9dc1b2ec9b43837b6d6d7c77431aaf873ba3c45dbe316fa55af87f0e` |
| `top20_milan_leonard_55774239` | `artifacts/top20_live_tapes80_20260826/tape_99566769_s1.py` | `caf952fb35080020a52e839950a4913d30557b248d2b153afbc46ef7282d9c3b` |
| `top20_arda_ceylan_55742738` | `artifacts/top20_live_tapes80_20260826/tape_99582725_s0.py` | `8e2801c8c1c4292dc5679207e722a548d84933d9caac128c2bdbedb848583915` |

## Equality and deliberate-difference checks

Machine comparison against the pinned V86-A protocol confirmed exact equality for action IDs and sources, all eight family objects, model, cross-validation, adequacy gates, selector gates, and promotion-grade gates.

The deliberate V86-B differences are limited to:

- fresh disjoint seeds `27100400..27100407`;
- the pre-outcome route-alias mechanical truth table and complete trajectory-equality contract;
- global nondegeneracy of at least 116/128 differing contexts for each of B and C, plus at least 12/16 differing contexts independently for each action in every family, counted solely from qualifying candidate executed-action differences;
- explicit exclusion of every V86-A economic margin and label.

The critic-strengthened mechanical contract additionally freezes:

- full joint prefix equality across KEEP/B/C before the branch: candidate raw/proposed/executed and opponent proposed/executed actions for steps 0..119, canonical observation trajectories for both seats, canonical engine-state trajectories through the pre-action step-120 state, and feature snapshots at 24/48/72/96/pre-action-120; every canonical payload length and SHA-256 digest must match or the experiment blocks;
- exact action-to-runtime-route mapping `FORCE_B120_SUFFIX -> b120` and `FORCE_C120_SUFFIX -> c120`; uppercase `B120`/`C120` are not runtime route keys;
- KEEP route choice absent through step 119, selected organically exactly once at step 120, then stable with zero drift through completion and no forced injection;
- exactly one forced injection at step 120, exactly one start and completion, requested route equal to selected route, no forced choice before step 120, and zero route-choice drift;
- `same KEEP route + identical full trajectory` as the only valid alias; `same route + different trajectory` blocks; a different selected route with an identical candidate executed trace also blocks;
- `VALID_NON_ALIAS` only when the candidate executed-action trace differs at least once and its earliest executed-action difference is step 120 or later; any earlier executed-action difference blocks;
- full equality over candidate raw/proposed/executed traces, opponent proposed/executed traces, canonical observation and state trajectories for both seats, canonical final state, both money values and signed margin, and feature snapshots at steps 24/48/72/96/120 for alias validation;
- raw-only, proposed-only, opponent-only, observation-only, state-only, feature-only, or route-state-only differences cannot establish non-alias support or contribute to the 116/128 and 12/16 counts;
- retention of alias rows as zero-paired-delta labels; selector choice of such a row still counts as an intervention and therefore fails the strictly-positive intervention gate;
- deterministic-core coverage of every joint-prefix payload/length/digest/verdict, route state and injection evidence, all traces, both-seat engine-state/observation trajectories, final state/money/margin, features, labels, selector accounting, and gate verdicts.

Any other contract difference, source-hash drift, forbidden V86-A dataset/result open or parse, joint KEEP/B/C prefix mismatch, runtime-route mapping miss, incomplete alias equality evidence, KEEP/forced route or injection drift, candidate executed difference before step 120, invalid post-branch executed-trace support count, or missed frozen gate is a stop condition. This receipt authorizes protocol implementation only; it does not authorize games, a checkpoint, sealed-holdout access, a submission, or a Kaggle upload.
