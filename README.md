# Binary-JEPA

> **Hachage Sémantique de Binaires par Prédiction Latente Auto-Supervisée sur Bag-of-Paths VEX**  
> Working Paper — ING3 Cybersécurité, CY Tech, 2026

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-rubenanko%2Fbinary--jepa-yellow)?style=for-the-badge](https://huggingface.co/rubenanko/binary-jepa)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)](https://www.python.org/)

---

## Vue d'ensemble

Binary-JEPA est une architecture de **hachage sémantique de binaires ELF64** fondée sur
l'apprentissage auto-supervisé. Face à l'obsolescence des signatures cryptographiques classiques
(MD5, SHA-256) face aux malwares polymorphes, ce projet propose de produire une empreinte
**stable face aux transformations compilatoires** et, à terme, aux techniques d'obfuscation modernes.

Le pipeline repose sur trois contributions articulées :

1. **Extraction sémantique** — lifting des instructions assembleur vers VEX IR via `angr`,
   modélisation topologique en _Bag-of-Paths_ et canonicalisation en 371 tokens abstraits par
   une projection stricte Φ
2. **I-JEPA 1D** — adaptation unidimensionnelle du cadre Joint-Embedding Predictive Architecture
   à l'analyse de séquences symboliques : l'encodeur apprend à _prédire_ la représentation latente
   de positions masquées plutôt qu'à reconstruire le token exact
3. **`elf64ctx`** — paquet Python distribuable encapsulant le pipeline complet, avec téléchargement
   automatique des artefacts depuis HuggingFace Hub

> ⚠️ **Working paper** : les hypothèses de résilience face aux obfuscations CFF/BCF restent à
> valider empiriquement. L'évaluation quantitative (Recall@k, MRR, AUC) est en cours.

---

## Structure du dépôt

```
binary-jepa/
├── latentSpace/
│   └── elf64-context-hash/          # Paquet Python elf64ctx
│       ├── elf64_context_hash/
│       │   ├── model/
│       │   │   ├── encoder.py       # Conv1DEncoder (fθ, fξ)
│       │   │   └── predictor.py     # Prédicteur gϕ
│       │   ├── cli.py               # Point d'entrée elf64ctx
│       │   ├── elf_processing.py    # Pipeline d'extraction VEX / Bag-of-Paths
│       │   ├── loaders.py           # Chargement vocab + checkpoint
│       │   └── constants.py
│       └── pyproject.toml
├── experiments/
│   ├── 0-firstOverview/             # Expérimentation qualitative (12 ELF, 66 paires)
│   └── 1-visualization/             # Visualisation multi-panels du pipeline
├── reports/
│   └── paper/                       # Working paper (LaTeX)
└── README.md
```

---

## Installation

```bash
pip install -e latentSpace/elf64-context-hash/
```

Le vocabulaire (`vocab.json`) et le checkpoint (`latest.pt`) sont téléchargés automatiquement
depuis [`rubenanko/binary-jepa`](https://huggingface.co/rubenanko/binary-jepa) au premier import.

**Dépendances principales** : `torch`, `angr`, `networkx`, `pebble`, `tqdm`

---

## Utilisation

### Encoder un binaire ELF

```bash
elf64ctx -E ./mon_binaire.elf
```

Produit `output/mon_binaire.elf-embeddings.json` — un dictionnaire
`{nom_fonction: [embeddings_b64, ...]}`.

### Comparer deux binaires

```bash
# Distances euclidiennes entre toutes les paires de fonctions
elf64ctx -C binaire_A.elf-embeddings.json binaire_B.elf-embeddings.json

# Avec heatmap
elf64ctx -C binaire_A.elf-embeddings.json binaire_B.elf-embeddings.json --plot

# Comparer deux fonctions spécifiques
elf64ctx -C binaire_A.elf-embeddings.json binaire_B.elf-embeddings.json \
         -F main main
```

### Mode prédicteur

```bash
# Utilise la sortie de gϕ plutôt que fθ (représentation prospective)
elf64ctx -E ./mon_binaire.elf -P
```

---

## Architecture

```
Binaire ELF
    │
    ▼
angr CFGFast ──► Graphe Gf = (Vf, Ef)
    │
    ▼
DFS aléatoire borné (Nmax=500, Lmax=50 blocs)
    │
    ▼
Projection Φ : VEX IR → 371 tokens canoniques
    │              Ist_Put → VEX_REG_WRITE
    │              call puts → <API_PUTS>
    │              jump → JK_BORING
    ▼
Bag-of-Paths : {p1, p2, ..., pN} séquences 1D
    │
    ▼
Conv1DEncoder fθ  (e=128, h=256, d=256, k=5/5/3)
    │
    ▼
Embedding ∈ ℝ²⁵⁶  (normalisé ℓ₂)
```

L'entraînement suit le paradigme **I-JEPA** : l'encodeur de contexte `fθ` prédit la
représentation latente de positions masquées, guidé par un encodeur cible `fξ` mis à jour
exclusivement par EMA (m=0.996). La perte de Huber (δ=1) est calculée uniquement sur les
positions cibles.

---

## Artefacts

Les artefacts sont hébergés sur HuggingFace 🤗 [`rubenanko/binary-jepa`](https://huggingface.co/rubenanko/binary-jepa) :

| Artefact        | Description                                             |
| --------------- | ------------------------------------------------------- |
| `latest.pt`     | Checkpoint de référence (run `random-token`, époque 10) |
| `vocab.json`    | Vocabulaire canonique (371 tokens)                      |
| `random-token/` | Run convergent — masquage aléatoire, L=0.030            |
| `last-token/`   | Run instabilité tardive — masquage dernier token        |
| `10-03-2026/`   | Run divergent — masquage par tiers                      |

---

## Résultats préliminaires

Validation qualitative sur 12 binaires GNU Coreutils
(gcc/clang × {-O0, -O3, -Os}, strip --strip-all) :

| Type de paire                                      | Distances observées | Interprétation                        |
| -------------------------------------------------- | ------------------- | ------------------------------------- |
| Positive (même programme, compilateurs différents) | [0.0, 1.0]          | Invariance partielle à la compilation |
| Négative (programmes différents, même compilateur) | [0.6, 1.0]          | Séparabilité inter-programmes         |
| Mixte (programmes + compilateurs différents)       | [0.0, 3.0]          | Effet super-additif                   |

---

## Feuille de route

- [ ] Évaluation quantitative — Recall@k, MRR, AUC sur les 1 060 ELF
- [ ] Validation empirique de la résilience BoP face à CFF/BCF (OLLVM)
- [ ] Analyse des clusters par projection UMAP/t-SNE
- [ ] Ablation des stratégies de masquage (_curriculum masking_)
- [ ] Extension de l'architecture (Mamba / Longformer) pour champ réceptif global
- [ ] Quantification INT8 + mode _fingerprint_ binaire (distance de Hamming O(1))
- [ ] Extension du corpus (noyau Linux, OpenSSL, FFmpeg) + fine-tuning sur malwares

---

## Citation

```bibtex
@techreport{bouchou2026binaryjepa,
  title   = {Binary-{JEPA} : Hachage S{\'e}mantique de Binaires par
             Pr{\'e}diction Latente Auto-Supervis{\'e}e sur Bag-of-Paths {VEX}},
  author  = {Bouchou, Ryan and Petteng-Ngongang, Ruben},
  year    = {2026},
  month   = {mars},
  type    = {Working Paper},
  institution = {CY Tech},
  note    = {\url{https://gitlab.com/iRyann/binary-jepa}}
}
```

---

## Auteurs

**Ryan Bouchou** — [ryanbouchou.fr](https://ryanbouchou.fr) · [ORCID](https://orcid.org/0009-0001-5714-3438)  
**Ruben Petteng-Ngongang**

ING3 Cybersécurité — CY Tech, Pau
