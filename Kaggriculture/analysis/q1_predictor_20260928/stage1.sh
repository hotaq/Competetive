#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
X=analysis/q1_predictor_20260928; mkdir -p $X/s1
R() { [ -f $3 ] && return; perl -e 'alarm shift; exec @ARGV' 1200 uv run python analysis/public_agents_20260924/play.py $1 "$2" $3.tmp >/dev/null 2>&1 && mv $3.tmp $3; echo "$3 $(date +%H:%M)"; }
for d in $X/variants/*/; do v=$(basename $d); R $d/main.py "vs|artifacts/public_notebooks_20260927/outputs/kaggriculture-ttv1/main.py|27700001|20" $X/s1/${v}_h2h.json; R $d/main.py replays:analysis/final_round_20260927/select $X/s1/${v}_sel.json; done
echo ALLDONE
