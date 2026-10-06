set -e
typeset -A OPP; OPP=(A artifacts/public_notebooks_20260924/outputs/a-song-of-ice-and-fire-fixed-flexible/main.py D artifacts/public_notebooks_20260924/outputs/kaggriculture-late-purchase-v16-submit/main.py E artifacts/public_notebooks_20260924/outputs/kaggriculture-v15stack-submit/main.py F artifacts/public_notebooks_20260924/outputs/the-2965-master-hybrid-engine/main.py G artifacts/public_notebooks_20260924/outputs/kaggriculture-thomas-2944-candidate/x/main.py X7 artifacts/public_notebooks_20260924/outputs/kaggriculture-shop-router-reactive-v7/main.py X5 artifacts/public_notebooks_20260924/outputs/kaggriculture-shop-router-reactive-v5/main.py Y tape:artifacts/tape_pool_20260922_top10/tapes/yannik_schiffner__ep111952809__seat0.json)
PLAY() { uv run python analysis/public_agents_20260924/play.py analysis/mirror_breaker_20260924/variants_r2/H54c/main.py "$@"; }
PLAY "vs|artifacts/public_notebooks_20260924/outputs/kaggriculture-harvest-ledger/main.py|25400001|100" analysis/mirror_breaker_20260924/confirm_r2/K1_vs_C.json | tail -1
PLAY replays:analysis/public_agents_20260924/live_56513871 analysis/mirror_breaker_20260924/confirm_r2/K2_Clive.json | tail -1
PLAY yannik analysis/mirror_breaker_20260924/confirm_r2/K2_yannik.json | tail -1
PLAY omar analysis/mirror_breaker_20260924/confirm_r2/K2_omar.json | tail -1
for k in A D E F G X7 X5 Y; do PLAY "vs|${OPP[$k]}|25000001|50" analysis/mirror_breaker_20260924/confirm_r2/K3_$k.json | tail -1 | sed "s#^.*: n=#K3 $k n=#"; done
