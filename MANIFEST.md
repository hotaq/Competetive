# Competetive

Code, notebooks, docs and small result files from the Kaggle / Super AI competition folders (`Super AI/competetive`), archived 2026-10-06.

**Not included** (too large for GitHub, regenerable, or sensitive): raw competition data, model checkpoints, zip/tgz/whl archives, `.venv`, `node_modules`, caches, files over the size cap (code/notebooks 5 MB, json/csv/txt 1 MB), and anything holding an API key or token.

Rules, applied identically to every folder: kept `.py .ipynb .md .yaml .toml .sh .js .html ...` and `.json/.csv/.txt` up to 1 MB; dropped every other file type.

| folder | files kept | MB kept | files left out | GB left out |
|---|---:|---:|---:|---:|
| RSNA | 1817 | 49.0 | 854 | 5.12 |
| Electric | 332 | 6.7 | 446 | 4.98 |
| Kaggriculture | 12507 | 1162.0 | 6194 | 2.82 |
| NeuroGolf2026 | 2520 | 368.9 | 8939 | 1.69 |
| Irrigation | 33 | 0.4 | 110 | 1.44 |
| Smartphone | 91 | 1.1 | 79 | 0.88 |
| Autonomous_Agent_Prediction | 746 | 150.3 | 205 | 0.13 |
| ROGII | 138 | 26.9 | 18 | 0.15 |
| Health_Risk | 1 | 0.0 | 8 | 0.12 |
| PlaygroundS6E5F1PitStops | 19 | 4.4 | 6 | 0.10 |
| superai6-iot-attack-detection | 102 | 9.0 | 7 | 0.02 |
| AI_Agent_Security | 310 | 1.4 | 45 | 0.02 |
| ARCPrize2026ARCAGI2 | 24 | 3.9 | 10 | 0.00 |
| NvidiaNemotronModelReasoning | 43 | 1.2 | 2 | 0.00 |
| OrbitWars | 48 | 4.0 | 1 | 0.00 |
| flowchart-to-code-alignment-verifier | 54 | 0.0 | 89 | 0.00 |
| MazeCrawler | 20 | 0.9 | 2 | 0.00 |

## Notes

- `Kaggriculture` was its own git repo (43 commits on `research/loss-and-market-analysis-20260830`, 107 uncommitted files, no remote). It is flattened in here **without its commit history**.
- `Electric/realmlp-td-s_standalone` is a clone of https://github.com/dholzmueller/realmlp-td-s_standalone and is not copied; clone it again if needed.
- Left out because they contain credentials: `OrbitWars/kaggle (1).json`, `PlaygroundS6E5F1PitStops/references/discussion_page.html`, `PlaygroundS6E5F1PitStops/references/discussion_696012.html`, `AI_Agent_Security/data/raw/kaggle/sdk/aicomp_sdk/fixtures/file_seed/api_keys.txt`; `.env` files and `.omx/` folders were never copied.
- Raw data and checkpoints must be re-downloaded from Kaggle.
