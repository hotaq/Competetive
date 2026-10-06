#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
F=analysis/final_round_20260927; mkdir -p $F/pub
typeset -A Q; Q=(Q1 artifacts/public_notebooks_20260927/outputs/kaggriculture-ttv1/main.py Q2 artifacts/public_notebooks_20260927/outputs/the-2965-master-hybrid-engine/main.py Q3 artifacts/public_notebooks_20260927/outputs/demand-preserving-turn-sale-timing/main.py Q4 artifacts/public_notebooks_20260927/outputs/farmer-john-and-the-idle-seller/main.py Q5 artifacts/public_notebooks_20260927/outputs/farmer-john-and-the-wheat-seller/v58_agent/main.py Q6 artifacts/public_notebooks_20260927/outputs/the-shepherds-ledger-herd-safe-sovereign/main.py Q7 artifacts/public_notebooks_20260927/outputs/lucky-boy-best-version-score-2552-1/built/main.py)
MS=submissions/n3_sb682_midswitch_standalone/main.py
R() { [ -f $3 ] && return; perl -e 'alarm shift; exec @ARGV' 900 uv run python analysis/public_agents_20260924/play.py $1 "$2" $3.tmp >/dev/null 2>&1 && mv $3.tmp $3; echo "$3 $(date +%H:%M)"; }
for k in Q7 Q1 Q2 Q3 Q4 Q5 Q6; do R ${Q[$k]} replays:$F/select $F/pub/${k}_sel.json; R ${Q[$k]} "vs|$MS|27400001|20" $F/pub/${k}_h2h.json; done
echo ALLDONE
