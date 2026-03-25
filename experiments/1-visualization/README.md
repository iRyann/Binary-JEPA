# Expérience 1 — Visualisation des stades de la pipeline

## Objectif

Visualiser côte à côte les 4 stades de transformation d'une fonction binaire :
**Assembleur → CFG+VEX IR → Chemins DFS (tokens) → IDs encodés**

L'outil produit une figure statique (PNG) multi-panels via
`binary-jepa/src/visualization/pipeline_viz.py`.

## Corpus

Binaires `cat` coreutils, compilés avec 3 profils :

| Profil | Compilateur | Optimisation |
|---|---|---|
| `cat_gcc_O0` | GCC | -O0 (aucune) |
| `cat_gcc_O3` | GCC | -O3 (agressive) |
| `cat_clang_Os` | Clang | -Os (taille) |

ELF sources : `experiments/0-firstOverview/data/binaries/`

## Cas étudiés

### 1. `cat_gcc_O0` → `io_blksize` (0x402d41)
**Pipeline complète, verbosité O0**
- 16 BBs, 24 arêtes, 165 instructions ASM
- 60 chemins DFS disponibles (16 affichés), moy. 40 tokens
- Ratio ASM→VEX : 0.73x — les instructions O0 sont peu denses en opérations
- H̄ = 2.75 bits — diversité sémantique modérée

### 2. `cat_gcc_O3` → `xpalloc` (0x409a70)
**Référence GCC -O3 pour l'étude comparative**
- 19 BBs, 27 arêtes, 70 instructions ASM
- 84 chemins DFS disponibles (16 affichés)
- Ratio ASM→VEX : 1.81x — O3 compacte l'assembleur mais le VEX IR reste expressif
- H̄ = 2.58 bits

### 3. `cat_clang_Os` → `xpalloc` (0x405dbd)
**Même sémantique, compilateur différent**
- 12 BBs, 16 arêtes, 70 instructions ASM
- 20 chemins DFS disponibles (16 affichés)
- Ratio ASM→VEX : 4.29x — clang -Os produit des BBs très compacts
- H̄ = 2.89 bits

## Observations

**Contraction syntaxique** : entre O0 et O3, le nombre de BBs augmente (16→19)
mais la densité informationnelle par instruction diminue — le VEX IR normalise
ces différences en produisant un flux de tokens comparable.

**Conservation sémantique** (cas 2 vs 3) : `xpalloc` compilé par GCC -O3 (19 BBs)
et Clang -Os (12 BBs) présentent des structures CFG très différentes, mais les
patterns de couleur dans le panel `CHEMINS DFS` sont quasi-identiques — les mêmes
catégories de tokens (REG, ARITH, MEM) apparaissent dans les mêmes proportions.
L'entropie de Shannon confirme cette proximité sémantique (2.58 vs 2.89 bits).

## Reproductibilité

```bash
cd binary-jepa/binary-jepa

# 1. Générer les JSONL (si absents)
python ../elf-processing/src/elf_processing_core.py \
    ../experiments/0-firstOverview/data/binaries/ 4 \
    experiments/1-visualization/data/

# 2. Encoder avec le vocab existant
python -c "
from src.preprocessing.vocab import encode_dataset
encode_dataset('experiments/1-visualization/data/',
               'experiments/1-visualization/encoded_dataset/',
               'vocab.json')
"

# 3. Régénérer les figures
python -m visualization.pipeline_viz \
    ../experiments/0-firstOverview/data/binaries/cat_gcc_O0.elf 0x402d41 \
    --jsonl-dir experiments/1-visualization/data/ \
    --encoded-dir experiments/1-visualization/encoded_dataset/ \
    --vocab vocab.json \
    --out experiments/1-visualization/output/cat_gcc_O0__io_blksize.png \
    --max-paths 16 --max-len 40

python -m visualization.pipeline_viz \
    ../experiments/0-firstOverview/data/binaries/cat_gcc_O3.elf 0x409a70 \
    --jsonl-dir experiments/1-visualization/data/ \
    --encoded-dir experiments/1-visualization/encoded_dataset/ \
    --vocab vocab.json \
    --out experiments/1-visualization/output/cat_gcc_O3__xpalloc.png \
    --max-paths 16 --max-len 40

python -m visualization.pipeline_viz \
    ../experiments/0-firstOverview/data/binaries/cat_clang_Os.elf 0x405dbd \
    --jsonl-dir experiments/1-visualization/data/ \
    --encoded-dir experiments/1-visualization/encoded_dataset/ \
    --vocab vocab.json \
    --out experiments/1-visualization/output/cat_clang_Os__xpalloc.png \
    --max-paths 16 --max-len 40
```
