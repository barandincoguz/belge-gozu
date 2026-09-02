# Refreshed Hugging Face index publication

Publication date: **2026-09-02**  
Dataset: `barandincoguz/belge-gozu-index`  
Immutable revision: `700ac324fffefb22de02c8e90347b31185547948`  
Superseded remote revision: `283a4c23bc32148b0ee96bab40e92664e7d2f2ea`

## What was published

Only the refreshed `index/` tree was uploaded. The existing 4,222 `images/*.webp` files
were counted before and after publication and were not re-uploaded. The new revision has
exactly these index files:

- `codes.npy`
- `manifest.json`
- `meta.parquet`
- `offsets.npy`
- `page_ids.json`
- `page_texts.parquet`
- `scales.npy`

The superseded `tokens.npy` and `page_vecs.npy` files are absent. Public Hub API resolution
of the pinned revision returned the same 40-character SHA.

## Manifest identity

| Field | Published value |
|---|---|
| model | `vidore/colSmol-500M` |
| model revision | `650243e9bf299a5a082841ed2907da8b0b9ce553` |
| query format | `train-compat-v1` |
| document prompt SHA-256 | `3d11cdfb8bca21c81671b3d074f446b3de06904fe98d184dec3a4c3e096b5212` |
| quantization | `int8` |
| mask policy | `drop-padding` |
| corpus checksum | `133444d8c235fb45795875c924ff44b6e33da80727ce9299741b4321982b8e9a` |
| pages | `4222` |
| tokens | `3759994` |
| render | `dpi=150`, `webp`, quality `80` |
| built at | `2026-08-27T16:09:09.004093+00:00` |
| source git commit recorded by manifest | `eb66be5` |

Before upload, `check_compatibility` returned no problems and the ordered page IDs in
`meta.parquet`, `page_ids.json`, and `page_texts.parquet` matched exactly at 4,222 rows.

## Fresh pinned-pull verification

The public artifact was downloaded without credentials into a newly created temporary
directory through `pull_index(..., revision=<SHA>, require_page_texts=True)`. Staged
manifest, checksum, representation signature, metadata order, and text alignment validation
all ran before the atomic destination swap. The temporary directory was then removed.

Result:

```text
pinned_pull=ok sha=700ac324fffefb22de02c8e90347b31185547948 checksum=133444d8c235fb45795875c924ff44b6e33da80727ce9299741b4321982b8e9a pages=4222 quantization=int8 aligned=True
```

No credential value or local `.env` content is recorded in this evidence.
