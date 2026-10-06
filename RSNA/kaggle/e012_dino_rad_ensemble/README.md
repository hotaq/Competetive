# E012 — DINO + self-trained Rad ensemble

Primary candidate:

```text
70% rank(E010 DINOv2 + DINOv3 fold-rank)
+ 30% mean(rank(E011 Rad fold))
```

Each Rad fold probability-averages two independently trained seeds before
ranking. Baker's cyst and Fracture remain on the DINO rank.

OOF evidence versus the exact DINO proxy:

- canonical fold macro: `0.852630 -> 0.859458`
- Gold58 macro: `0.843676 -> 0.860123`
- global cross-fit alpha: mean `0.29`

The encoder is the official MIT RadImageNet checkpoint. The ten heads are our
E011 outputs; the public non-commercial head package is not attached.
