#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
X=analysis/midswitch_20260926
for d in $X/variants/*/; do v=$(basename $d); f=$X/holdout/$v.json; [ -f $f ] && continue
  uv run python analysis/public_agents_20260924/play.py $d/main.py replays:analysis/rl_endgame_20260926/holdout/replays $f.tmp >/dev/null 2>&1 && mv $f.tmp $f; echo "$v $(date +%H:%M)"; done
echo ALLDONE
