#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
X=analysis/route_rl_20260926
for v in native R105 R106 R107 R120 R125; do f=$X/holdout/$v.json; [ -f $f ] && continue; uv run python analysis/public_agents_20260924/play.py $X/variants/$v/main.py replays:analysis/rl_endgame_20260926/holdout/replays $f.tmp >/dev/null 2>&1 && mv $f.tmp $f; echo "$v $(date +%H:%M)"; done
echo ALLDONE
