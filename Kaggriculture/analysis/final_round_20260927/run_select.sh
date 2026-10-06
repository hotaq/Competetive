#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
F=analysis/final_round_20260927; mkdir -p $F/sel
R() { [ -f $2 ] && return; uv run python analysis/public_agents_20260924/play.py $1 replays:$F/select $2.tmp >/dev/null 2>&1 && mv $2.tmp $2; echo "$2 $(date +%H:%M)"; }
R submissions/n3_sb682_midswitch_standalone/main.py $F/sel/MS.json
for d in $F/variants/*/; do R $d/main.py $F/sel/$(basename $d).json; done
echo ALLDONE
