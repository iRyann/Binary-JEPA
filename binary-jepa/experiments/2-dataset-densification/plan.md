# Dataset Densification Plan — Binary-JEPA

**Status**: evaluated, sources verified (2026-07-27). Nothing downloaded yet.
**Motivation**: Stage B (`data_v3/stats_dedup.json`) revealed the corpus bottleneck —
100 Coreutils × 10 toolchain variants contain only **2.2M tokens of unique semantic
content** after anti-leakage dedup (gnulib sharing across programs, 10 trivial
families 100% boilerplate). More variants of the *same* programs cannot fix this;
we need more *programs*, more *architectures*, and *obfuscation transforms*.

**Free invariance signal**: VEX IR is architecture-agnostic — a cross-arch corpus
gives same-function/different-arch pairs for free through the existing angr pipeline.

---

## 1. Verified sources

### Tier 1 — Direct fit (highest priority)

#### 1.1 BinKit 2.0 (KAIST, SoftSec) — best single match
- **Content**: 371,928 pre-compiled ELF binaries from GNU packages (coreutils,
  busybox, openssl, gsl…), 1,904 build configurations:
  - 8 architectures: x86-32/64, ARM-32/64 (LE), MIPS-32/64 (LE + BE)
  - 6 optimization levels: `-O0 -O1 -O2 -O3 -Os -Ofast`
  - 23 compilers: gcc 4.9.4→11.2.0, clang 4.0.0→13.0.0
  - **`clang-obfus` variants** (Obfuscator-LLVM: `fla`, `sub`, `bcf`, `all`)
    → obfuscation-resilience training pairs *with ground truth*, no obfuscator
    toolchain to run ourselves
- **Ground truth**: sister tool TikNib (BCSA ground-truth builder); function
  identity known by construction (same package × config matrix)
- **License**: MIT (code); binaries inherit source licenses (GPL-family)
- **Access**: pre-compiled dataset + full build scripts (crosstool-ng)
- **Repo**: https://github.com/SoftSec-KAIST/BinKit
- **Paper**: Kim et al., "Revisiting Binary Code Similarity Analysis…",
  IEEE TSE 2022, doi:10.1109/TSE.2022.3187689
- **Caveat**: toolchain targets Ubuntu 16.04-era compilers — irrelevant for us,
  old compilers are diversity, not staleness. Tested with Python 3.8.

#### 1.2 Assemblage — build farm + published corpus
- **Content**: distributed corpus generator; scrapes license-filtered C/C++
  GitHub repos, builds with **gcc/clang × `-O0..-Os`** (our exact variant matrix),
  archives binaries with **DWARF function-level metadata** (names/addresses/line
  info → identity keys + probe construction). Also Rust (3 codegen backends).
- **Deliverables**: docker-compose farm (MIT) + published permissively-licensed
  SQLite corpus at https://assemblage-dataset.net
- **Repo**: https://github.com/Assemblage-Dataset/Assemblage
- **Paper**: arXiv:2405.03991
- **Use**: published corpus for volume now; clone the farm later for
  purpose-built variants.

#### 1.3 BinaryCorp (jTrans, ISSTA'22)
- **Content**: binaries mined from **Arch Linux official repos + AUR** — real-world,
  non-GNU-farm diversity (most diverse BCSD corpus to date, per the paper).
  Raw binaries + preprocessed features publicly released.
- **Repo**: https://github.com/vul337/jTrans (dataset links in README)
- **Paper**: Wang et al., ISSTA 2022, doi:10.1145/3533767.3534367
- **Caveat**: no per-binary build metadata (community compilation) → unlabeled
  volume, not ground truth. Use for style diversity, not for probe construction.

### Tier 2 — Source-level (compile ourselves, identity keys included)

| Source | Content | Why |
|---|---|---|
| **AnghaBench** (UFMG) | **1M compilable C functions** mined from major GitHub C repos (curl, git, openssl, FFmpeg, linux…), single-file compilable | Function-level scale; compile under our 2×5 matrix → identity-keyed pairs for free |
| **ExeBench** (PLDI'22) | **4.5M compilable / 700k executable C functions** from GitHub, external types resolved | Largest compilable-function corpus; executable subset enables future dynamic validation |
| **More GNU packages** | binutils, findutils, diffutils, grep/sed/gawk, tar, util-linux, procps | Trivial extension of the existing farm, +30–50 program families |

- AnghaBench: https://github.com/brenocfg/AnghaBench
- ExeBench: Armengol-Estapé et al., "ExeBench: an ML-scale dataset of executable
  C functions", PLDI 2022, doi:10.1145/3520312.3534867
- Source-level corpora give `function_identity_key` (repo + file + function name)
  → powers the Stage E semantic probe and validates LSH clusters vs ground truth.

### Obfuscation transforms

| Tool | Status | Use |
|---|---|---|
| **OLLVM** (via BinKit `clang-obfus`) | ships inside BinKit 2.0 | v1 of the obfuscation axis, zero extra work |
| **Tigress** (U. Arizona) | active, academic standard | source-level: control-flow flattening, virtualization, opaque predicates, literal encoding; seeded transforms on AnghaBench/ExeBench subsets → unlimited pairs with perfect ground truth. https://tigress.wtf |
| **Hikari** | ⚠️ **DEPRECATED** (archived 2023-09, author advises against; fork `NeHyci/Hikari-LLVM15`) | **do not use** — OLLVM + Tigress cover the need |
| **UPX / packers** | — | **eval-only** transform: packing breaks static CFG assumptions differently than compiler obfuscation → separate OOD test tier |

### Eval-domain sources (OOD test tiers, NOT pre-training)

- **MalwareBazaar** (abuse.ch): free API, daily samples, ELF coverage
  (computes TrendMicro `telfhash` for ELF). Small ELF subset = "in-the-wild
  obfuscated" eval tier. Safety: static analysis only, quarantined storage,
  never execute, never redistribute. https://bazaar.abuse.ch
- **Alpine Linux packages**: musl libc → libc-implementation diversity
  (entire current corpus is glibc — hidden bias). Debian ports: prebuilt
  multi-arch ELFs.

---

## 2. Target composition (post-dedup, desugared tokens)

| Slice | Source | Role | Est. post-dedup |
|---|---|---|---|
| Core farm | Current Coreutils ×10 | seed | 2.2M (done: `data_v3/`) |
| GNU + arch + obfus | BinKit 2.0 subset (~20–40k binaries) | volume + invariance pairs | 50–150M |
| Real-world | BinaryCorp | style diversity | 30–80M |
| Function-level | AnghaBench/ExeBench compiled subset (50–100k files × 2–3 variants) | scale + identity keys | 50–150M |
| Obfuscated | BinKit `clang-obfus` + Tigress-seeded | resilience pairs | 20–50M |
| **Total** | | | **~150–400M tokens** |

Raw volumes are 10–40× larger before Stage A+B compression. **The real
constraint is extraction throughput**: angr CFG on ~50k binaries is a
cluster job, not a laptop job (see §5, step 0).

---

## 3. Schema — extend the v3 record with provenance

```json
{
  "corpus_id": "binkit2.0",
  "package": "openssl-1.1.1", "family": "openssl",
  "arch": "arm_64", "compiler": "gcc-9.4.0", "opt": "O2",
  "obfuscator": "ollvm-fla", "obf_seed": null,
  "function_key": "sha256_block_data_order",
  "func_addr": "0x1a3f0", "path_id": 12, "cluster_id": 88123,
  "split": "train", "tokens": ["..."]
}
```

- `corpus_id` ∈ `coreutils | binkit2.0 | assemblage | binarycorp | angha | exebench`
- `function_key` from DWARF/symbols when available, else `null`
- `obfuscator`/`obf_seed` null for non-obfuscated records

## 4. Layout & pipeline integration

```
data_v4/
  <corpus_id>/<family>/<family>_<compiler>_<opt>[_<obf>].jsonl   ← shards
  manifest.parquet          ← one row per record, all metadata (fast splits/stats)
  stats_dedup.json
```

- **One global LSH across all corpora** (Stage B rerun on the union): gnulib-style
  sharing exists *between* corpora too (openssl in BinKit ∩ AnghaBench's openssl
  files). Global `cluster_id` keeps the anti-leakage invariant corpus-wide.
- **Recalibrate `--boilerplate-min-families`**: with thousands of families, N=50
  becomes too loose → switch to a fraction (e.g., ≥10% of families) and re-tune.
- **Three-tier splits** (hashed on `cluster_id` / `function_key`):
  1. *In-distribution*: random cluster split (current mechanism);
  2. *Cross-variant*: hold out an entire axis value (all `mips_64`, all
     `ollvm-bcf`) → arch/obfuscation generalization;
  3. *OOD*: hold out entire `family` sets → Allamanis-safe generalization test.
- **Versioning**: content hash (SHA-256) per shard in the manifest; corpora are
  append-only, never edited in place.
- **Licenses**: keep `corpus_id` provenance so a permissively-licensed-only
  redistribution subset can be derived later (Assemblage already publishes
  exactly such a subset).

---

## 5. Order of operations

0. **Extraction farm design** (prerequisite): batch angr over a BinKit-style
   directory tree — worker pool, per-binary timeout, resumable state (skip
   existing shards), failure log. Design before pulling any source.
1. **BinKit 2.0 normal + obfus subset** — highest value/effort ratio
   (pre-built, arch + obfuscation axes, ground truth).
2. **Re-run Stage A+B on the union** — validates pipeline scaling and
   re-quantifies unique content. **This is the go/no-go for pre-training size.**
3. **AnghaBench subset through our own farm** — identity-keyed volume.
4. **BinaryCorp** (unlabeled volume), then **MalwareBazaar/Alpine** as eval
   tiers once the Stage E probe exists.

---

## 6. Risks & caveats

- **Allamanis duplication returns at corpus scale**: shared libraries (openssl,
  zlib, gnulib) appear in *every* source. Global LSH + function-bag excision
  is the mitigation; expect the post-dedup compression ratio to stay brutal
  (v1: 40×) — plan extraction volume accordingly.
- **angr coverage on non-x86**: MIPS/ARM lifting is supported by VEX but less
  exercised; validate CFG quality on a small arch sample before mass extraction.
- **Obfuscated binaries and CFG recovery**: `bcf`/`fla` inflate basic blocks and
  defeat heuristics — expect higher angr failure/timeout rates on the obfus
  slice; that's acceptable (log and skip), but monitor the failure rate per
  transform so eval conclusions aren't skewed by extraction bias.
- **Malware handling**: static-only, quarantined directory, no execution,
  no redistribution; check institutional policy before pulling.
- **Storage**: BinKit 2.0 full is large; pull the *subset* matching our matrix
  (2 compilers × 5 opts × 2–3 archs + obfus) rather than the full 371k binaries.

---

*Verified against primary sources (2026-07-27): BinKit README, jTrans repo +
ISSTA'22 bibtex, Assemblage README + arXiv:2405.03991, AnghaBench repo,
ExeBench PLDI'22 (Semantic Scholar), tigress.wtf, Hikari deprecation notice,
bazaar.abuse.ch.*
