N3=artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py
H=submissions/mirror_h54c_standalone/main.py
while read fam name spec; do
  for arm in N3 H54c; do
    [ $arm = N3 ] && me=$N3 || me=$H
    out=analysis/generalization_20260924/runs/${arm}__${fam}_${name}.json; [ -f $out ] && continue
    uv run python analysis/public_agents_20260924/play.py $me "vs|$spec|25700001|30" $out 2>&1 | tail -1 | sed "s#^.*: n=#$arm $fam $name n=#"
  done
done < analysis/generalization_20260924/opponents.txt
# F3 cross arms: N3 vs H54c and H54c vs N3 already covered by M1 on other seeds; play them here too on the same grid
uv run python analysis/public_agents_20260924/play.py $N3 "vs|$H|25700001|30" analysis/generalization_20260924/runs/N3__F3_H54c.json | tail -1 | sed "s#^.*: n=#N3 F3 H54c n=#"
