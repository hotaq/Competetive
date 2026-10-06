typeset -A C; C=(P1 artifacts/public_notebooks_20260926/outputs/kaggriculture-harvest-ledger/main.py P2 artifacts/public_notebooks_20260926/outputs/kaggriculture-multi-route-farming-agent/main.py P3 artifacts/public_notebooks_20260926/outputs/kaggriculture-top-2-master-engine-v4/main.py P4 artifacts/public_notebooks_20260926/outputs/kaggriculture-hack/main.py P5 artifacts/public_notebooks_20260926/outputs/the-shepherds-ledger-herd-safe-sovereign/main.py P6 artifacts/public_notebooks_20260926/outputs/kaggriculture-version-31-26-09-bronze-going-up/main.py)
for k in P2 P4 P3 P1 P5 P6; do
  for pn in "vs|artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py|26100001|20" "vs|submissions/n3_sb682_standalone/main.py|26100001|20" replays:analysis/n3_live_20260925/replays replays:analysis/public_agents_20260924/live_56513871; do
    case $pn in *n3_sb682*) tag=vsSB;; vs*) tag=vsN3;; *n3_live*) tag=N3live;; *) tag=Clive;; esac
    uv run python analysis/public_agents_20260924/play.py ${C[$k]} "$pn" analysis/public_agents_20260926/screen/${k}_${tag}.json 2>&1 | tail -1 | sed "s#^.*: n=#$k $tag n=#"
  done
done
