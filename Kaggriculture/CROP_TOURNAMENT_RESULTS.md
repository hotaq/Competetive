# Crop Heuristic Tournament Results

ผลนี้เป็น local Kaggle-score proxy จาก Rust engine ที่ผ่าน oracle parity ไม่ใช่คะแนนจาก
Kaggle leaderboard โดยตรง ทุก matchup ใช้ weed rate มาตรฐาน `0.005`, เล่นครบ episode,
และสลับทั้ง player 0/player 1 เพื่อลด seat bias

## Recommended baseline

ใช้ `MELON:6:r0` เป็น deterministic crop baseline เริ่มต้น:

```python
CropHeuristicPolicy(crop="MELON", target_quantity=6, sell_reserve=0)
```

เหตุผล: เป็น configuration เดียวที่ชนะทุกเกมใน diverse gauntlet รอบสุดท้าย และยังชนะ
MELON-heavy meta บน seed ชุดใหม่อย่างชัดเจน

## Diverse gauntlet

- 30 configurations: crop ทั้ง 5 ชนิด × plots `1,2,4,6,8,12`
- final seeds: `901,1009,1103`
- 720 configured episode steps
- แต่ละ configuration มี 174 seat-balanced results

| Rank | Config | W/L | Win rate | Mean money | Worst money | P10 margin | Worst margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `MELON:6:r0` | 174/0 | 100.0% | 16,325.8 | 15,043 | +5,152.5 | +212 |
| 2 | `MELON:12:r0` | 164/10 | 94.3% | 16,074.7 | 13,462 | +4,864.4 | -1,626 |
| 3 | `MELON:8:r0` | 166/8 | 95.4% | 15,970.5 | 14,422 | +4,594.4 | -1,083 |
| 4 | `MELON:4:r0` | 156/18 | 89.7% | 12,067.1 | 11,666 | -627.2 | -3,788 |
| 5 | `MELON:2:r0` | 150/24 | 86.2% | 7,810.4 | 7,558 | -6,633.9 | -9,096 |

## MELON-heavy stress test

- MELON plots `6,7,8,9,10,11,12,13,14,15,16,18,20`
- final seeds: `401,503,607,709,811`
- `MELON:6` won 117/120, mean money 15,070.2, P10 margin +346,
  worst margin -307

ผลนี้แสดงว่าปลูกมากเกินไปกดราคาตลาดเมื่อคู่แข่งขาย MELON เหมือนกัน จึงทำให้ 6 plots
ทนต่อ mirror/counter-meta มากกว่า 8 หรือ 12 plots

## Top-three fresh-seed audit

- candidates: MELON plots `6,8,12`
- 16 seeds ที่ไม่ใช้ในรอบเลือก configuration
- 64 seat-balanced results ต่อ configuration

| Config | W/L | Win rate | Mean money | Worst money | P10 margin | Worst margin |
|---|---:|---:|---:|---:|---:|---:|
| `MELON:6:r0` | 61/3 | 95.3% | 15,082.1 | 15,027 | +267.8 | -307 |
| `MELON:8:r0` | 30/34 | 46.9% | 14,572.8 | 14,371 | -666.0 | -790 |
| `MELON:12:r0` | 5/59 | 7.8% | 14,383.9 | 13,302 | -934.0 | -1,786 |

## Reproduce

Broad two-stage search:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_crop_configs.py \
  --output results/crop_tournament.json
```

Focused fresh-seed audit:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_crop_configs.py \
  --crops MELON --plots 6,8,12 --screen-seeds 43 --screen-steps 120 \
  --finalists 3 \
  --final-seeds 1201,1213,1229,1249,1277,1291,1303,1327,1361,1381,1409,1423,1451,1471,1499,1523 \
  --final-steps 720 --workers 4 --output results/melon_top3_16seed.json
```
