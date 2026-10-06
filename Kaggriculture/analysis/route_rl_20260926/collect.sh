#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
X=analysis/route_rl_20260926
for d in $X/variants/*/; do v=$(basename $d); for p in n3live:analysis/n3_live_20260925/replays sb682:analysis/n3_clone_detect_20260926/sb682_panel/replays clive:analysis/public_agents_20260924/live_56513871 yannik:analysis/live_56495539_20260924/replays omar:analysis/live_56495539_20260924/confirm_56361130/replays; do tag=${p%%:*}; f=$X/train/${v}_${tag}.json; [ -f $f ] && continue; uv run python analysis/public_agents_20260924/play.py $d/main.py replays:${p#*:} $f.tmp >/dev/null 2>&1 && mv $f.tmp $f; done; echo "$v done $(date +%H:%M)"; done
echo ALLDONE
