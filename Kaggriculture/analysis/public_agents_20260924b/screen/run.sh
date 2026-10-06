typeset -A N; N=(N1 artifacts/public_notebooks_20260924b/outputs/kaggriculture-harvest-ledger/main.py N2 artifacts/public_notebooks_20260924b/outputs/kaggriculture-population-robust-economy/main.py N3 artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py N4 artifacts/public_notebooks_20260924b/outputs/kaggriculture-autonomous-ai-farming-agent/main.py N5 artifacts/public_notebooks_20260924b/outputs/herd-safe-v3-experimental-risk-aware-feed/main.py N6 artifacts/public_notebooks_20260924b/outputs/kaggriculture-master-engine-v3/main.py N7 artifacts/public_notebooks_20260924b/outputs/kaggriculture-herd-safe-sale-window-race-ca25/x/main.py N8 artifacts/public_notebooks_20260924b/outputs/the-shepherds-ledger-herd-safe-sovereign/main.py N9 artifacts/public_notebooks_20260924b/outputs/kaggriculture-v40-challenger/main.py)
for k in N1 N2 N3 N4 N5 N6 N7 N8 N9; do
  for pn in "vs|submissions/mirror_h54c_standalone/main.py|25500001|20" yannik omar replays:analysis/public_agents_20260924/live_56513871; do
    tag=$(echo $pn | cut -c1-8 | tr ':|/' '___')
    uv run python analysis/public_agents_20260924/play.py ${N[$k]} "$pn" analysis/public_agents_20260924b/screen/${k}_${tag}.json 2>&1 | tail -1 | sed "s#^.*: n=#$k $tag n=#"
  done
done
