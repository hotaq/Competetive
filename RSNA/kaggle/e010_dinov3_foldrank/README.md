# E010 — DINOv3 fold-rank probe

Controlled Kaggle-only inference experiment:

```text
0.65 * rank(scored DINOv2 0.899 family)
+ 0.35 * mean(rank(each DINOv3 fold))
```

The DINOv3 implementation and checkpoints come from the public
[`romantamrazov/rsna-knee-dinosaur-v2`](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2)
pipeline and its credited upstream contributors. The source notebook is byte
pinned in `kaggle/dinov3_public_anchor/`.

This probe intentionally excludes RadImageNet and all target/sample-adaptive
hidden-test weighting. It writes:

- `submission_public_0899.csv`: preserved DINOv2 safety artifact;
- `submission_e010_d3_probmean.csv`: historical probability-mean diagnostic;
- `submission_e010_d3_foldrank.csv`: controlled candidate;
- `submission.csv`: byte-equivalent candidate used by Kaggle submission;
- `e010_manifest.json`: UID/schema/finiteness/hash evidence.
