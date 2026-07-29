# binary-jepa — Context for Claude Code

## Project Overview

**binary-jepa** is a research project applying JEPA (Joint Embedding Predictive Architecture)
to binary analysis. It:
1. Loads ELF binaries via **angr** (CFG reconstruction, VEX IR lifting)
2. Tokenizes VEX IR basic blocks and extracts **DFS paths** through the CFG
3. Encodes token sequences with a vocabulary (`vocab.json`, ~370 tokens)
4. Trains a JEPA model to learn latent representations of binary execution paths

## Repository Layout

```
binary-jepa/
├── CLAUDE.md                          ← you are here
├── binary-jepa/                       ← Python package root
│   ├── src/
│   │   ├── visualization/             ← pipeline visualization (matplotlib)
│   │   │   ├── pipeline_viz.py        ← CLI entry-point, data orchestration
│   │   │   ├── asm_panel.py           ← Panel 1: disassembly dump
│   │   │   ├── cfg_panel.py           ← Panel 2: CFG with VEX tokens in nodes
│   │   │   ├── paths_panel.py         ← Panel 3: DFS paths as colored token grid
│   │   │   ├── ids_panel.py           ← Panel 4: encoded IDs heatmap
│   │   │   ├── token_colors.py        ← token→category→color mapping
│   │   │   ├── theme.py               ← dark-mode color constants
│   │   │   └── base_panel.py          ← abstract Panel base class
│   │   ├── models/                    ← JEPA model definition
│   │   ├── preprocessing/             ← tokenization pipeline
│   │   │   ├── desugar.py             ← Stage A: VEX noise removal (streaming)
│   │   │   └── dedup.py               ← Stage B: anti-leakage dedup (MinHash LSH)
│   │   └── masks/                     ← masking strategies
│   ├── data/                          ← training dataset v1 (1000 JSONL files, ~318M tokens)
│   ├── data_v2/                       ← desugared dataset (182M tokens, 0% structural noise)
│   ├── data_v3/                       ← deduplicated dataset (~25k paths, cluster_id + split)
│   ├── encoded_dataset/               ← encoded training data (int IDs)
│   ├── experiments/
│   │   ├── 0-dataset-evaluation/      ← streaming dataset diagnostics
│   │   │   ├── evaluate.py            ← global diagnostic v1 (streaming, no RAM overflow)
│   │   │   ├── evaluate_v2.py         ← diagnostic v2 + v1↔v2 comparison table
│   │   │   ├── report.txt             ← v1 report (raw dataset)
│   │   │   └── report_v2.txt          ← v2 report (desugared dataset)
│   │   ├── 1-visualization/           ← per-binary visualization examples
│   │   ├── 2-dataset-densification/   ← corpus expansion plan (verified sources)
│   │       ├── data/                  ← subset JSONL for 12 binaries
│   │       ├── encoded_dataset/       ← encoded subset
│   │       └── output/                ← generated PNGs
│   └── vocab.json                     ← {token: id} for 370 tokens
└── .venv/                             ← virtual environment (Python 3.12)
```

## Environment

**Always use the project venv** — system Python lacks angr, networkx, etc.:
```bash
/home/ryan/dev/binary-jepa/.venv/bin/python
```

## Key Commands

### Visualization (offline, no angr)

```bash
cd /home/ryan/dev/binary-jepa/binary-jepa

# Single combined figure
/home/ryan/dev/binary-jepa/.venv/bin/python -m src.visualization.pipeline_viz \
    /path/to/binary.elf 0x414d41 \
    --no-angr \
    --jsonl-dir     experiments/1-visualization/data/ \
    --encoded-dir   experiments/1-visualization/encoded_dataset/ \
    --vocab         vocab.json \
    --out           experiments/1-visualization/output/result.png

# Separate PNG per panel (recommended for large CFGs or many paths)
/home/ryan/dev/binary-jepa/.venv/bin/python -m src.visualization.pipeline_viz \
    /path/to/binary.elf 0x414d41 \
    --no-angr --separate \
    --jsonl-dir     experiments/1-visualization/data/ \
    --encoded-dir   experiments/1-visualization/encoded_dataset/ \
    --vocab         vocab.json \
    --out           experiments/1-visualization/output/result.png
```

Good example functions with real path diversity:
- `ls_gcc_O0.jsonl` → `0x414d41`  (many BBs, diverse DFS paths)
- `ls_clang_Os.jsonl` → `0x40e6f3`

### Desugaring (Stage A — VEX noise removal)

```bash
cd /home/ryan/dev/binary-jepa/binary-jepa
/home/ryan/dev/binary-jepa/.venv/bin/python -m src.preprocessing.desugar \
    --data-dir data/ --out-dir data_v2/ --vocab-out vocab_v2.json
# Flags: --no-fold-width (drop casts without fusing width into ops),
#        --run-cap N (max run length for REG_READ/REG_WRITE/CONST, default 2),
#        --keep-abihint (retain VEX_Ist_AbiHint)
```

Rewrite grammar: drops `VEX_WrTmp` + `Ist_{Dirty,MBE,PutI,AbiHint}`, folds type-cast
tokens into the preceding op (`VEX_OP_CMP + VEX_OP_1UT → VEX_OP_CMP.u1`), drops orphan
casts, caps identical-token runs at 2. Results (report_v2.txt): 318M → 182M tokens
(-42.7%), structural noise 33.5% → 0%, median path 263 → 155 tokens, p99 1433 → 734.
Note: per-token Shannon entropy *decreases* (3.53 → 3.20 bits) — expected: removed casts
had a flat tail that inflated v1 entropy. Signal vs diversity must be judged on the
downstream semantic probe, not on Shannon entropy.

### Deduplication (Stage B — anti-leakage)

```bash
cd /home/ryan/dev/binary-jepa/binary-jepa
/home/ryan/dev/binary-jepa/.venv/bin/python -m src.preprocessing.dedup \
    --data-dir data_v2/ --out-dir data_v3/
# Flags: --keep-per-cluster K (max representatives per LSH cluster, default 2),
#        --boilerplate-min-families N (function-bag excision threshold, default 50),
#        --val-pct P (cluster-level val split, default 5)
```

Three levels: (1) exact blake2b dedup, (2) MinHash (128 perms, 5-gram shingles)
+ LSH (16 bands) clustering with exact Jaccard ≥ 0.85 validation, keeping ≤ K
representatives per cluster preferring *distinct compiler variants*, (3) boilerplate
excision of whole function bags whose path-set fingerprint appears in ≥ N binary
families. Split is hashed on `cluster_id` → zero train/val leakage (verified:
0 mixed clusters). Output schema: `{file, func_addr, path_id, cluster_id, split, tokens}`.

Results (stats_dedup.json): 950k → 24.8k paths (2.2M tokens), 23.4k clusters,
426k boilerplate paths dropped, residual near-dup 11.3% (deliberate K=2 variant pairs).
**Key finding**: the corpus's unique semantic content is far smaller than raw volume
suggested — gnulib utility code is shared verbatim across coreutils families
(cross-family clusters like b2sum↔df confirm it). 10 trivial families (true, sync,
whoami…) are 100% boilerplate and vanish; 90 families remain (median 117 paths/family).
If 2.2M tokens proves too small for pre-training, the knobs are `--keep-per-cluster 3-4`
or corpus augmentation (more programs / obfuscated variants).

### Dataset Evaluation

```bash
cd /home/ryan/dev/binary-jepa/binary-jepa
/home/ryan/dev/binary-jepa/.venv/bin/python \
    experiments/0-dataset-evaluation/evaluate.py
# Streams all 1000 JSONL in data/, writes experiments/0-dataset-evaluation/report.txt

/home/ryan/dev/binary-jepa/.venv/bin/python \
    experiments/0-dataset-evaluation/evaluate_v2.py
# Same metrics on data_v2/ + v1↔v2 comparison → report_v2.txt
```

## Dataset Characteristics (as of 2026-03)

Key findings from the global evaluation (1000 binaries, 318M tokens):

| Metric | Value |
|---|---|
| Total paths | 950 474 |
| **Global duplicates** | **74.6%** — cross-binary duplicates (same function across compiler variants) |
| Distinct tokens in vocab | 370 |
| Shannon entropy | 3.53 / 8.53 bits (41% vocab utilization) |
| Top 5 tokens | 67.8% of corpus |
| VEX_REG_WRITE alone | 27.4% of corpus |
| Median path length | 263 tokens |
| Median distinct tokens/path | 16 |

**Implication**: DFS paths share long common prefixes (function preamble). Divergence typically
occurs after position ~50. When visualizing, the `--max-len` window must exceed 50 or the
common-prefix trimmer will show the divergence zone automatically.

## Visualization Architecture

- Each panel is a standalone `Panel` subclass (`render(ax)` method, `title`, `subtitle` props)
- `pipeline_viz.py` loads all data and instantiates panels — panels have no angr dependency
- Layout uses `ax.transAxes` (normalized [0,1] coords), not `ax.transData` — panels are
  resolution-independent and clip gracefully
- `--separate` mode saves each panel as an individual PNG sized to content:
  - ASM: 0.13" per disassembly line
  - CFG: driven by `n_layers × 2.5"` and `max_per_layer × 3.5"`
  - Paths/IDs: `n_paths × 0.38"` + margins
- `_trim_common_prefix()` in `pipeline_viz.py` automatically trims shared preamble tokens
  before visualizing DFS paths (keeps 6 tokens of context before first divergence point)

## Token Coloring

Tokens are colored in two layers (`token_colors.py`):
1. **Exact match** (`_TOKEN_COLORS`): manually chosen high-contrast colors for the 20 most
   important tokens (REG_READ/WRITE, LOAD/STORE, JK_*, EXIT_COND, ...)
2. **Hash-based** (`_category_token_color`): deterministic color within the category's hue
   family for unknown tokens

Categories: REG, ARITH, MEM, CTRL, FLOW, TEMP, API, SYSCALL, SPECIAL

## Known Issues / Future Work

- **High duplicate rate (74.6%)**: same function from `ls_gcc_O0` vs `ls_gcc_O3` produces
  near-identical paths. Consider deduplication or cross-binary contrastive sampling for JEPA.
- **VEX IR verbosity**: structural tokens (WrTmp, type-casts) = 33.5% of corpus and carry
  little semantic signal. Consider filtering or down-weighting them.
- **Short effective vocabulary**: top 20 tokens = 98.9% of corpus. The model sees highly
  repetitive sequences.
- **angr mode** (`--no-angr` absent): not tested in this session; requires a working angr
  install and a real ELF binary. The offline path is the tested code path.
