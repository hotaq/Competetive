#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
X=analysis/react_dump_20260926
for v in RD30s RD50s RD30a RD50a; do for p in n3live:analysis/n3_live_20260925/replays sb682:analysis/n3_clone_detect_20260926/sb682_panel/replays clive:analysis/public_agents_20260924/live_56513871 yannik:analysis/live_56495539_20260924/replays omar:analysis/live_56495539_20260924/confirm_56361130/replays; do tag=${p%%:*}; f=$X/train/${v}_${tag}.json; [ -f $f ] && continue; uv run python analysis/public_agents_20260924/play.py $X/variants/$v/main.py replays:${p#*:} $f.tmp >/dev/null 2>&1 && mv $f.tmp $f; done; echo "$v $(date +%H:%M)"; done
echo ALLDONE
