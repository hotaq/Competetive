#!/bin/zsh
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
F=analysis/final_round_20260927
ARGS=($(cat $F/fresh_args.txt))
uv run --with kaggle python scripts/harvest_loss_replays.py $ARGS --output-dir $F/replays > $F/harvest.log 2>&1
echo ALLDONE >> $F/harvest.log
