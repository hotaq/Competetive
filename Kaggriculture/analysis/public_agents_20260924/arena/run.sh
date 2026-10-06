set -e
typeset -A M; M=(
 C artifacts/public_notebooks_20260924/outputs/kaggriculture-harvest-ledger/main.py
 A artifacts/public_notebooks_20260924/outputs/a-song-of-ice-and-fire-fixed-flexible/main.py
 D artifacts/public_notebooks_20260924/outputs/kaggriculture-late-purchase-v16-submit/main.py
 E artifacts/public_notebooks_20260924/outputs/kaggriculture-v15stack-submit/main.py
 F artifacts/public_notebooks_20260924/outputs/the-2965-master-hybrid-engine/main.py
 G artifacts/public_notebooks_20260924/outputs/kaggriculture-thomas-2944-candidate/x/main.py
 X7 artifacts/public_notebooks_20260924/outputs/kaggriculture-shop-router-reactive-v7/main.py
 X5 artifacts/public_notebooks_20260924/outputs/kaggriculture-shop-router-reactive-v5/main.py
 Y tape:artifacts/tape_pool_20260922_top10/tapes/yannik_schiffner__ep111952809__seat0.json
)
K=(C A D E F G X7 X5 Y)
for ((i=1;i<=${#K};i++)); do for ((j=i+1;j<=${#K};j++)); do a=${K[i]}; b=${K[j]}
  [ $a = Y ] && continue
  out=analysis/public_agents_20260924/arena/${a}_vs_${b}.json; [ -f $out ] && continue
  uv run python analysis/public_agents_20260924/play.py ${M[$a]} "vs|${M[$b]}|25000001|50" $out | tail -1 | sed "s#.*: n=#$a vs $b n=#"
done; done
