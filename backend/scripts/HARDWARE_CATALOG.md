# Download a GPU hardware catalogue

From the repository root, with Python 3.10+ (no pip packages required):

```bash
python backend/scripts/download_hardware_catalog.py
```

This downloads the GPU JSON and licence from
[gmasse/gpu-specs](https://github.com/gmasse/gpu-specs), validates pinned SHA-256
checksums over HTTPS, and converts the data to Compute Cast's existing schema.
The reviewed source version is pinned to commit
`578a5d7c9d148158822069ce2b43bab0133ff306`; this is a reproducible snapshot,
not an automatically updated complete inventory of modern GPUs.

Default outputs:

- `backend/data/hardware_catalog.downloaded.json`
- `backend/data/hardware_catalog.downloaded.LICENSE.txt`

The existing `backend/data/hardware_catalog.json` is read for merging and is
**not modified**. Exact matching curated records win as whole entries, preserving
their specifications and verification metadata. The default download currently
produces 15 entries, including the two retained curated GPU records.

To choose a different filename, for example a separate `hardware_catalog.json`:

```bash
python backend/scripts/download_hardware_catalog.py --output hardware_catalog.json
```

To regenerate the separate download, explicitly allow replacing it:

```bash
python backend/scripts/download_hardware_catalog.py --force
```

Use `--no-merge` to generate only community records, or `--merge path/to/curated.json`
to preserve another catalogue. The script refuses to use the merge input as its
output. Keep the licence sidecar with redistributed generated data.

## Converted fields

Each GPU includes name/manufacturer, architecture, form factor when explicit,
advertised memory capacity, memory type/bandwidth, FP32/FP64/FP16 scalar throughput,
and explicitly supplied dense FP16/BF16/TF32 Tensor Core throughput. Source links,
retrieval date, dataset commit/checksum and verification status are retained.

- No pricing, cloud credentials, local measurements or GPU execution are involved.
- Sparse throughput is never substituted for dense throughput or divided by two
  to invent a missing dense specification. Unknown or unsupported peaks are null.
- Multi-capacity entries such as `16/32` retain those options and leave the chosen
  capacity null. The script does not guess the exact card variant.
- GB and GB/s remain as labelled upstream. Advertised GB is not asserted to be
  exact allocatable VRAM in GiB. Confirm memory units/capacity for your target.
- Community imports are **not independently verified**. Their `last_verified`
  field is null; reference links and a datasheet-review status are provided.
  Some entries are incomplete or inconsistent. Review them against NVIDIA's
  exact-variant datasheet before using them for decisions.
- An imported H100 or H200 NVL entry must not automatically be equated with a
  Nebius H100 SXM or H200 SXM platform. This script does not create those mappings.

Review the separate generated file before incorporating entries into the curated
catalogue. The script does not change the app's hardware selection endpoints or
estimator formulas. A technical GPU catalogue also does not describe machine
allocation, CPU RAM, storage or effective communication throughput; those belong
in a separate target-machine configuration.

When updating the pinned dataset, review the upstream commit, schema, units and
licence, then record checksums for that version. Never disable checksum or TLS
verification to work around a changed source.
