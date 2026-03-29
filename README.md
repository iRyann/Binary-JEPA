# Binary-JEPA

> **Hachage Sémantique de Binaires par Prédiction Latente Auto-Supervisée sur Bag-of-Paths VEX**  
> Working Paper — ING3 Cybersécurité, CY Tech, 2026

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-rubenanko%2Fbinary--jepa-yellow)](https://huggingface.co/rubenanko/binary-jepa)
[![Working Paper](https://img.shields.io/badge/paper-working%20paper-lightgrey)](reports/paper/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

## Vue d'ensemble

Binary-JEPA est une architecture de **hachage sémantique de binaires ELF64** fondée sur l'apprentissage auto-supervisé. Face à l'obsolescence des signatures cryptographiques classiques (MD5, SHA-256) face aux malwares polymorphes, ce projet propose de produire une empreinte **stable face aux transformations compilatoires** et, à terme, aux techniques d'obfuscation modernes.

### Pipeline

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 140" font-family="monospace" font-size="12">
  <!-- fond -->
  <rect width="860" height="140" fill="#0d1117" rx="8"/>

  <!-- étape 1 : ELF -->
  <rect x="20" y="30" width="120" height="80" rx="6" fill="#161b22" stroke="#30363d" stroke-width="1.2"/>
  <text x="80" y="60" text-anchor="middle" fill="#8b949e" font-size="10">Binaire</text>
  <text x="80" y="76" text-anchor="middle" fill="#e6edf3" font-weight="bold">ELF64</text>
  <text x="80" y="94" text-anchor="middle" fill="#8b949e" font-size="10">x86 · ARM</text>

  <!-- flèche 1 -->
  <line x1="140" y1="70" x2="175" y2="70" stroke="#30363d" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="157" y="63" text-anchor="middle" fill="#8b949e" font-size="9">angr</text>

  <!-- étape 2 : VEX IR -->
  <rect x="175" y="30" width="130" height="80" rx="6" fill="#161b22" stroke="#3fb950" stroke-width="1.2"/>
  <text x="240" y="55" text-anchor="middle" fill="#8b949e" font-size="10">Lifting +</text>
  <text x="240" y="71" text-anchor="middle" fill="#3fb950" font-weight="bold">VEX IR</text>
  <text x="240" y="87" text-anchor="middle" fill="#8b949e" font-size="10">canonicalisation Φ</text>
  <text x="240" y="100" text-anchor="middle" fill="#6e7681" font-size="9">371 tokens</text>

  <!-- flèche 2 -->
  <line x1="305" y1="70" x2="340" y2="70" stroke="#30363d" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="322" y="63" text-anchor="middle" fill="#8b949e" font-size="9">DFS</text>

  <!-- étape 3 : Bag-of-Paths -->
  <rect x="340" y="30" width="140" height="80" rx="6" fill="#161b22" stroke="#58a6ff" stroke-width="1.2"/>
  <text x="410" y="55" text-anchor="middle" fill="#8b949e" font-size="10">Bag-of-Paths</text>
  <text x="410" y="71" text-anchor="middle" fill="#58a6ff" font-weight="bold">séquences 1D</text>
  <text x="410" y="87" text-anchor="middle" fill="#8b949e" font-size="10">N ≤ 500 chemins</text>
  <text x="410" y="100" text-anchor="middle" fill="#6e7681" font-size="9">L ≤ 50 blocs</text>

  <!-- flèche 3 -->
  <line x1="480" y1="70" x2="515" y2="70" stroke="#30363d" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="497" y="63" text-anchor="middle" fill="#8b949e" font-size="9">I-JEPA</text>

  <!-- étape 4 : encodeur -->
  <rect x="515" y="30" width="150" height="80" rx="6" fill="#161b22" stroke="#ffa657" stroke-width="1.2"/>
  <text x="590" y="55" text-anchor="middle" fill="#8b949e" font-size="10">Conv1DEncoder</text>
  <text x="590" y="71" text-anchor="middle" fill="#ffa657" font-weight="bold">f_θ  (15 Mo)</text>
  <text x="590" y="87" text-anchor="middle" fill="#8b949e" font-size="10">e=128 · h=256 · d=256</text>
  <text x="590" y="100" text-anchor="middle" fill="#6e7681" font-size="9">k = 5 / 5 / 3</text>

  <!-- flèche 4 -->
  <line x1="665" y1="70" x2="700" y2="70" stroke="#30363d" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="682" y="63" text-anchor="middle" fill="#8b949e" font-size="9">ℓ₂</text>

  <!-- étape 5 : embedding -->
  <rect x="700" y="30" width="140" height="80" rx="6" fill="#161b22" stroke="#d2a8ff" stroke-width="1.2"/>
  <text x="770" y="55" text-anchor="middle" fill="#8b949e" font-size="10">Embedding</text>
  <text x="770" y="71" text-anchor="middle" fill="#d2a8ff" font-weight="bold">ℝ²⁵⁶</text>
  <text x="770" y="87" text-anchor="middle" fill="#8b949e" font-size="10">normalisé</text>

  <!-- marqueur flèche -->
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#30363d"/>
    </marker>
  </defs>
</svg>
```

---

## Structure du dépôt

```
binary-jepa/
│
├── binary-jepa/                    ← Package d'entraînement (Python 3.12)
│   ├── src/
│   │   ├── models/
│   │   │   ├── jepa.py             ← IJEPA (context encoder + target encoder + predictor)
│   │   │   ├── encoder.py          ← Conv1DEncoder (fθ, fξ)
│   │   │   └── predictor.py        ← Prédicteur gϕ (MLP 2 couches)
│   │   ├── masks/
│   │   │   ├── random_token.py     ← Masquage aléatoire ← run convergent
│   │   │   ├── last_token.py       ← Masquage dernier token
│   │   │   └── noiseproof.py       ← Masquage par tiers (objectif I-JEPA 1D)
│   │   ├── preprocessing/
│   │   │   ├── vocab.py            ← Construction du vocabulaire + encodage dataset
│   │   │   └── dataset.py          ← BagOfPathsDataset (PyTorch)
│   │   ├── train/
│   │   │   └── default.py          ← Boucle d'entraînement + codecarbon
│   │   ├── test/
│   │   │   ├── encoder.py          ← Évaluation cosine/distance par paires
│   │   │   └── predictor.py        ← Évaluation mode prédicteur
│   │   └── visualization/          ← 4 panels : ASM · CFG · DFS · IDs
│   │       ├── pipeline_viz.py     ← Orchestrateur CLI
│   │       ├── cfg_panel.py
│   │       ├── paths_panel.py
│   │       ├── ids_panel.py
│   │       └── token_colors.py     ← Catégorisation sémantique des tokens
│   ├── data/                       ← 1 000 JSONL · 950 474 chemins · 318 M tokens
│   ├── encoded_dataset/
│   ├── vocab.json                  ← 371 tokens canoniques
│   ├── train.yaml
│   └── train.sh                    ← Lance l'entraînement + génère le rapport CSV
│
├── elf-processing/                 ← Pipeline d'extraction dataset
│   ├── src/elf_processing_core.py  ← BinaryAnalyzer (angr + DFS + VEX)
│   ├── external/dataGen/
│   │   └── generate_data.sh        ← Compilation GNU Coreutils (gcc/clang × 5 niveaux)
│   └── pipeline.sh                 ← Orchestrateur complet
│
├── latentSpace/
│   └── elf64-context-hash/         ← Paquet distribuable elf64ctx
│       ├── elf64_context_hash/
│       │   ├── cli.py              ← Interface CLI (-E / -C / -P / --plot)
│       │   ├── elf_processing.py   ← BinaryAnalyzer (miroir elf-processing)
│       │   ├── loaders.py          ← Chargement vocab + checkpoint HuggingFace
│       │   ├── model/
│       │   │   ├── encoder.py
│       │   │   └── predictor.py
│       │   └── constants.py
│       └── pyproject.toml
│
├── experiments/
│   ├── 0-firstOverview/            ← 12 ELF · 66 paires · heatmaps qualitatifs
│   └── 1-visualization/            ← Visualisation multi-panels du pipeline
│
├── reports/paper/                  ← Working paper (LaTeX / arxiv.sty)
├── requirements.txt
└── README.md
```

---

## Dataset

Le dataset d'entraînement est généré automatiquement à partir de **GNU Coreutils 9.10** compilé en cross-compilation contrôlée :

```
C × O = {gcc, clang} × {-O0, -O1, -O2, -O3, -Os}
→ 106 utilitaires × 10 variants = 1 060 fichiers ELF (strip --strip-all)
```

| Métrique                          | Valeur           |
| --------------------------------- | ---------------- |
| Chemins totaux                    | 950 474          |
| Duplicats inter-binaires          | **74.6 %**       |
| Tokens distincts (vocab)          | 371              |
| Entropie de Shannon               | 3.53 / 8.53 bits |
| Top 5 tokens                      | 67.8 % du corpus |
| `VEX_REG_WRITE` seul              | 27.4 % du corpus |
| Longueur médiane / chemin         | 263 tokens       |
| Tokens distincts médians / chemin | 16               |

> **Note** : le taux de duplicats élevé (74.6 %) est attendu — deux compilations d'une même fonction partagent un long préfixe commun dans le CFG. La divergence significative n'apparaît généralement qu'après la position ~50. L'outil de visualisation gère cela via `_trim_common_prefix()`.

---

## Installation

### 1. Inférence uniquement — `elf64ctx`

```bash
pip install -e latentSpace/elf64-context-hash/
```

Le vocabulaire (`vocab.json`) et le checkpoint (`latest.pt`, ~15 Mo) sont téléchargés automatiquement depuis [`rubenanko/binary-jepa`](https://huggingface.co/rubenanko/binary-jepa) au premier import, dans `~/.elf64-context-hash/`.

**Dépendances** : `torch`, `angr`, `networkx`, `pebble`, `tqdm`, `huggingface_hub`

### 2. Entraînement — package complet

```bash
# Environnement virtuel recommandé (Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Dépendances système pour angr
pip install angr networkx pebble
```

---

## Utilisation

### Encoder un binaire ELF

```bash
elf64ctx -E ./mon_binaire.elf
# → output/mon_binaire.elf-embeddings.json
#   {nom_fonction: [embedding_b64, ...]}
```

### Comparer deux binaires

```bash
# Toutes les paires de fonctions
elf64ctx -C binaire_A.elf-embeddings.json binaire_B.elf-embeddings.json

# Avec heatmap (colormap plasma_r)
elf64ctx -C binaire_A.elf-embeddings.json binaire_B.elf-embeddings.json --plot

# Deux fonctions spécifiques
elf64ctx -C A.json B.json -F main main

# Mode prédicteur (sortie de gϕ à la position <MASK>)
elf64ctx -E ./mon_binaire.elf -P
```

### Visualiser le pipeline (offline, sans angr)

```bash
cd binary-jepa/binary-jepa

python -m src.visualization.pipeline_viz \
    /path/to/binary.elf 0x40e6f3 \
    --no-angr \
    --jsonl-dir     experiments/1-visualization/data/ \
    --encoded-dir   experiments/1-visualization/encoded_dataset/ \
    --vocab         vocab.json \
    --out           output/result.png \
    --separate          # un PNG par panel (recommandé pour les grands CFGs)
```

Fonctions avec bonne diversité de chemins :

- `ls_gcc_O0.jsonl` → `0x414d41` (beaucoup de BBs, chemins diversifiés)
- `ls_clang_Os.jsonl` → `0x40e6f3`

---

## Entraînement

```bash
cd binary-jepa/binary-jepa

# 1. Générer le dataset ELF
cd ../../elf-processing && bash pipeline.sh

# 2. Construire le vocabulaire
python -m src.preprocessing.vocab data/ --out vocab.json --min_freq 2

# 3. Encoder le dataset
python -m src.preprocessing.vocab data/ --with-vocab vocab.json

# 4. Sérialiser en raw_data
python -m src.preprocessing.vocab encoded_dataset/ --save-raw raw_data500.json

# 5. Lancer l'entraînement (+ rapport graphique automatique)
bash train.sh --config_path train.yaml
```

**Configuration** (`train.yaml`) :

```yaml
train_data_path: raw_data500.json
vocab_path: vocab.json
epoch: 10
batch_size: 10
learning_rate: 1e-4
use_checkpoint: false
checkpoint_path: latest.pt
```

L'empreinte carbone est tracée automatiquement via `codecarbon`.

---

## Historique des runs

Trois runs archivés sur [`rubenanko/binary-jepa`](https://huggingface.co/rubenanko/binary-jepa) :

| Run            | Stratégie masquage | LR     | Batch | Perte finale | Statut                 |
| -------------- | ------------------ | ------ | ----- | ------------ | ---------------------- |
| `10-03-2026`   | Tiers consécutifs  | 5×10⁻⁵ | 20    | 512.88       | ❌ divergence          |
| `last-token`   | Dernier token      | 10⁻⁴   | 10    | 5.81         | ⚠️ instabilité tardive |
| `random-token` | Token aléatoire    | 10⁻⁴   | 10    | **0.030**    | ✅ convergent          |

Le checkpoint de référence (`latest.pt`) correspond au run `random-token`, époque 10.

---

## Architecture I-JEPA 1D

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 320" font-family="monospace" font-size="11">
  <rect width="700" height="320" fill="#0d1117" rx="8"/>

  <!-- Séquence masquée -->
  <text x="160" y="28" text-anchor="middle" fill="#8b949e" font-size="10">séquence masquée x̃</text>
  <rect x="60"  y="35" width="40" height="26" rx="3" fill="#1f6feb33" stroke="#58a6ff" stroke-width="1"/>
  <text x="80"  y="52" text-anchor="middle" fill="#58a6ff" font-size="9">VEX</text>
  <rect x="105" y="35" width="40" height="26" rx="3" fill="#1f6feb33" stroke="#58a6ff" stroke-width="1"/>
  <text x="125" y="52" text-anchor="middle" fill="#58a6ff" font-size="9">OP</text>
  <rect x="150" y="35" width="40" height="26" rx="3" fill="#da363333" stroke="#f78166" stroke-width="1"/>
  <text x="170" y="52" text-anchor="middle" fill="#f78166" font-size="9">[M]</text>
  <rect x="195" y="35" width="40" height="26" rx="3" fill="#da363333" stroke="#f78166" stroke-width="1"/>
  <text x="215" y="52" text-anchor="middle" fill="#f78166" font-size="9">[M]</text>
  <rect x="240" y="35" width="40" height="26" rx="3" fill="#30363d55" stroke="#6e7681" stroke-width="1"/>
  <text x="260" y="52" text-anchor="middle" fill="#6e7681" font-size="9">[P]</text>

  <!-- Séquence intacte -->
  <text x="510" y="28" text-anchor="middle" fill="#8b949e" font-size="10">séquence intacte x</text>
  <rect x="410" y="35" width="40" height="26" rx="3" fill="#1f6feb33" stroke="#58a6ff" stroke-width="1"/>
  <text x="430" y="52" text-anchor="middle" fill="#58a6ff" font-size="9">VEX</text>
  <rect x="455" y="35" width="40" height="26" rx="3" fill="#1f6feb33" stroke="#58a6ff" stroke-width="1"/>
  <text x="475" y="52" text-anchor="middle" fill="#58a6ff" font-size="9">OP</text>
  <rect x="500" y="35" width="40" height="26" rx="3" fill="#3fb95033" stroke="#3fb950" stroke-width="1"/>
  <text x="520" y="52" text-anchor="middle" fill="#3fb950" font-size="9">API</text>
  <rect x="545" y="35" width="40" height="26" rx="3" fill="#3fb95033" stroke="#3fb950" stroke-width="1"/>
  <text x="565" y="52" text-anchor="middle" fill="#3fb950" font-size="9">JK</text>
  <rect x="590" y="35" width="40" height="26" rx="3" fill="#30363d55" stroke="#6e7681" stroke-width="1"/>
  <text x="610" y="52" text-anchor="middle" fill="#6e7681" font-size="9">[P]</text>

  <!-- Encodeur contexte -->
  <rect x="80" y="105" width="200" height="60" rx="6" fill="#1f6feb1a" stroke="#58a6ff" stroke-width="1.5"/>
  <text x="180" y="131" text-anchor="middle" fill="#58a6ff" font-weight="bold">Encodeur Contexte  fθ</text>
  <text x="180" y="150" text-anchor="middle" fill="#8b949e" font-size="10">Conv1D k=5/5/3 · GELU</text>

  <!-- Encodeur cible -->
  <rect x="420" y="105" width="200" height="60" rx="6" fill="#3fb9501a" stroke="#3fb950" stroke-width="1.5"/>
  <text x="520" y="131" text-anchor="middle" fill="#3fb950" font-weight="bold">Encodeur Cible  fξ</text>
  <text x="520" y="150" text-anchor="middle" fill="#8b949e" font-size="10">EMA  ξ ← m·ξ + (1-m)·θ</text>

  <!-- EMA flèche -->
  <path d="M280,125 C350,95 370,95 420,125" fill="none" stroke="#3fb950" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#ga)"/>

  <!-- Représentations -->
  <rect x="100" y="195" width="160" height="30" rx="4" fill="#1f6feb0d" stroke="#58a6ff" stroke-width="1"/>
  <text x="180" y="215" text-anchor="middle" fill="#58a6ff" font-size="10">Z_ctx ∈ ℝ^(L×256)</text>

  <rect x="440" y="195" width="160" height="30" rx="4" fill="#3fb9500d" stroke="#3fb950" stroke-width="1"/>
  <text x="520" y="215" text-anchor="middle" fill="#3fb950" font-size="10">Z_tgt ∈ ℝ^(L×256)</text>

  <!-- stop-grad -->
  <line x1="520" y1="195" x2="520" y2="188" stroke="#3fb950" stroke-dasharray="3,2" stroke-width="1"/>
  <text x="540" y="187" fill="#3fb95099" font-size="9">stop-grad</text>

  <!-- Prédicteur -->
  <rect x="260" y="255" width="180" height="50" rx="6" fill="#ffa6571a" stroke="#ffa657" stroke-width="1.5"/>
  <text x="350" y="278" text-anchor="middle" fill="#ffa657" font-weight="bold">Prédicteur  gϕ</text>
  <text x="350" y="295" text-anchor="middle" fill="#8b949e" font-size="10">Linear(256→512→256) · GELU</text>

  <!-- Flèches verticales -->
  <line x1="160" y1="61"  x2="160" y2="105" stroke="#58a6ff" stroke-width="1.2" marker-end="url(#ga)"/>
  <line x1="520" y1="61"  x2="520" y2="105" stroke="#3fb950" stroke-width="1.2" marker-end="url(#ga)"/>
  <line x1="180" y1="165" x2="180" y2="195" stroke="#58a6ff" stroke-width="1.2" marker-end="url(#ga)"/>
  <line x1="520" y1="165" x2="520" y2="195" stroke="#3fb950" stroke-width="1.2" marker-end="url(#ga)"/>
  <line x1="180" y1="225" x2="300" y2="255" stroke="#58a6ff" stroke-width="1.2" marker-end="url(#ga)"/>
  <line x1="520" y1="225" x2="400" y2="255" stroke="#3fb950" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#ga)"/>

  <!-- Huber Loss label -->
  <text x="350" y="318" text-anchor="middle" fill="#d2a8ff" font-size="10">Huber Loss sur M_pred uniquement</text>

  <defs>
    <marker id="ga" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#30363d"/>
    </marker>
  </defs>
</svg>
```

---

## Artefacts HuggingFace

| Artefact        | Description                                             |
| --------------- | ------------------------------------------------------- |
| `latest.pt`     | Checkpoint de référence (run `random-token`, époque 10) |
| `vocab.json`    | Vocabulaire canonique (371 tokens)                      |
| `random-token/` | ✅ Run convergent — plateau à L=0.030                   |
| `last-token/`   | ⚠️ Instabilité tardive à l'époque 8                     |
| `10-03-2026/`   | ❌ Divergence catastrophique (L=512 ep10)               |

---

## Résultats préliminaires

Validation qualitative sur 12 binaires GNU Coreutils  
(`gcc`/`clang` × {`-O0`, `-O3`, `-Os`}, `strip --strip-all`) :

| Type de paire                                      | Distances  | Interprétation                        |
| -------------------------------------------------- | ---------- | ------------------------------------- |
| Positive (même programme, compilateurs différents) | [0.0, 1.0] | Invariance partielle à la compilation |
| Négative (programmes différents, même compilateur) | [0.6, 1.0] | Séparabilité inter-programmes         |
| Mixte (programmes + compilateurs différents)       | [0.0, 3.0] | Effet super-additif                   |

---

## Limites connues

- **Taux de duplicats élevé (74.6 %)** — les fonctions partagent un long préfixe commun inter-compilateurs. Un échantillonnage contrastif inter-binaires serait plus efficace pour I-JEPA.
- **Verbosité VEX IR** — les tokens structurels (`WrTmp`, casts de types) représentent ~33.5 % du corpus avec peu de signal sémantique.
- **Vocabulaire effectif restreint** — les 20 premiers tokens couvrent 98.9 % du corpus. Les séquences sont très répétitives.
- **Champ réceptif Conv1D limité à 13 tokens** — incompatible avec la stratégie de masquage par tiers (objectif théorique de I-JEPA 1D). Un encodeur à champ global (Mamba, Longformer) est nécessaire.
- **Validation obfuscation absente** — la résilience face à CFF/BCF reste à valider empiriquement sur des binaires OLLVM.

---

## Feuille de route

- [ ] Recall@k · MRR · AUC sur les 1 060 ELF du corpus
- [ ] Validation CFF/BCF (OLLVM) — légitimation formelle du Bag-of-Paths
- [ ] Analyse UMAP/t-SNE des embeddings latents
- [ ] Ablation systématique des stratégies de masquage (_curriculum masking_)
- [ ] Extension architecture (Mamba/Longformer) pour champ réceptif global
- [ ] Quantification INT8 + mode _fingerprint_ (Hamming O(1))
- [ ] Extension corpus (noyau Linux, OpenSSL, FFmpeg) + fine-tuning malwares

---

## Citation

```bibtex
@techreport{bouchou2026binaryjepa,
  title       = {Binary-{JEPA} : Hachage S{\'e}mantique de Binaires par
                 Pr{\'e}diction Latente Auto-Supervis{\'e}e sur Bag-of-Paths {VEX}},
  author      = {Bouchou, Ryan and Petteng-Ngongang, Ruben},
  year        = {2026},
  month       = {mars},
  type        = {Working Paper},
  institution = {CY Tech},
  note        = {\url{https://gitlab.com/iRyann/binary-jepa}}
}
```

---

## Auteurs

**Ryan Bouchou** — [ryanbouchou.fr](https://ryanbouchou.fr) · [ORCID 0009-0001-5714-3438](https://orcid.org/0009-0001-5714-3438)  
**Ruben Petteng-Ngongang**

ING3 Cybersécurité — CY Tech, Pau, 2026
