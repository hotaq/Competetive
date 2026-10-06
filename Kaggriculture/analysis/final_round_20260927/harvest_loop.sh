#!/bin/zsh
# usage: harvest_loop.sh <argsfile> <logfile>; restarts the harvester every 90 s so a hung download costs <= 90 s
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture"
F=analysis/final_round_20260927
ARGS=($(cat $1))
for i in {1..60}; do
  n=$(/bin/ls $F/replays | grep -c '^ep.*json$')
  [ $n -ge 328 ] && break
  perl -e 'alarm shift; exec @ARGV' 90 uv run --with kaggle python scripts/harvest_loss_replays.py $ARGS --output-dir $F/replays >> $2 2>&1
done
echo ALLDONE >> $2
