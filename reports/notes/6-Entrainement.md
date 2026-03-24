### Section 3.5 : Stratégie d'Entraînement

---

#### 3.5.1 Architecture JEPA : Encodeur Contextuel et Encodeur Cible

L'entraînement suit fidèlement le paradigme **I-JEPA** (Image Joint-Embedding Predictive Architecture) transposé à la dimension temporelle des séquences de tokens VEX. Le système repose sur trois composants distincts :

- **L'encodeur contextuel** (`Conv1DEncoder`) reçoit la séquence d'exécution **partiellement masquée** et produit une représentation latente de dimension 256 pour chaque position.
- **L'encodeur cible** est une copie de l'encodeur contextuel, mise à jour **sans rétropropagation** par une **moyenne exponentielle mobile (EMA)** de momentum m = 0.996 :

  ```
  θ_cible ← 0.996 · θ_cible + 0.004 · θ_contextuel
  ```

  Il reçoit la séquence **originale non masquée** et fournit les représentations de référence — des cibles stables qui ne s'effondrent pas vers une solution triviale, contrairement à un entraînement symétrique.
- **Le prédicteur** (`Predictor`) — un MLP léger — prend en entrée la représentation contextuelle et prédit la représentation cible aux positions masquées.

L'encodeur cible est initialisé comme copie exacte de l'encodeur contextuel (momentum = 0 au premier appel), puis diverge progressivement au fil des mises à jour EMA pour constituer un signal de supervision lissé.

---

#### 3.5.2 Stratégie de Masquage : Last Token Mask

Le choix de la stratégie de masquage est central dans JEPA. Nous avons adopté le **Last Token Mask**, motivé par la nature causale et séquentielle des chemins d'exécution : un chemin est une trace ordonnée de blocs de base, où le contexte passé détermine la continuation.

Concrètement, pour une séquence de longueur L :
- Le **masque d'entrée** supprime les tokens de la position ⌊2L/3⌋ à L — le dernier tiers de la séquence est remplacé par le token `<MASK>` (ID : 2) avant passage dans l'encodeur contextuel.
- Le **masque de prédiction** cible uniquement la position finale (L − 1) — c'est la seule position sur laquelle la loss est calculée.

Cette asymétrie (masquer un tiers, prédire un seul token) pousse le modèle à **comprimer la sémantique de tout le préfixe visible** en une représentation capable d'anticiper l'état final du chemin, sans se laisser distraire par la reconstruction de tokens intermédiaires triviaux.

Quatre stratégies de masquage ont été implémentées et comparées (`last_token`, `random_token`, `random_block`, `noiseproof`) ; `last_token` a été retenue comme stratégie par défaut pour son alignement avec la structure causale des données.

---

#### 3.5.3 Fonction de Perte : Smooth L1 (Loss de Huber)

La loss est calculée exclusivement sur les positions prédites (masque de prédiction) :

```
L = Smooth_L1(z_pred[masque_pred], z_cible[masque_pred])
```

Le choix de la **Smooth L1 Loss** (dite Loss de Huber) par rapport à la MSE classique est délibéré : elle est quadratique pour les petites erreurs (comportement doux) et linéaire pour les grandes (robustesse aux outliers). Dans notre contexte, certaines fonctions produisent des chemins d'exécution atypiques — fonctions de gestion d'erreur, stubs minimaux — dont les représentations seraient des outliers pénalisants sous MSE.

---

#### 3.5.4 Architecture des Modules

**Conv1DEncoder** — Encodeur contextuel et cible (partagent la même architecture) :

| Couche | Opération | Entrée → Sortie |
|--------|-----------|-----------------|
| Embedding | `Embedding(371, 128)` | (B, L) → (B, L, 128) |
| Conv1 | `Conv1d(128, 256, k=5, p=2)` + GELU | (B, 128, L) → (B, 256, L) |
| Conv2 | `Conv1d(256, 256, k=5, p=2)` + GELU | (B, 256, L) → (B, 256, L) |
| Conv3 | `Conv1d(256, 256, k=3, p=1)` | (B, 256, L) → (B, 256, L) |

Les kernels de taille 5 capturent le contexte local sur une fenêtre de ±2 tokens ; le kernel final de taille 3 affine la représentation. Le padding `same` préserve la longueur de séquence à chaque couche.

**Predictor** — MLP de projection :

| Couche | Opération |
|--------|-----------|
| Linear | 256 → 512 |
| GELU | — |
| Linear | 512 → 256 |

Le prédicteur est intentionnellement **peu profond** : il doit projeter la représentation contextuelle vers l'espace cible sans court-circuiter l'apprentissage de l'encodeur. Un prédicteur trop expressif apprendrait à compenser les faiblesses de l'encodeur.

---

#### 3.5.5 Apprentissage Multi-Instances (Deep Sets sur le Bag-of-Paths)

Chaque fonction est représentée par jusqu'à **16 chemins d'exécution**, chacun padé à **64 tokens**. Le tenseur d'entrée au modèle est de forme `(Batch × Paths, L)` — les chemins d'une même fonction sont traités **indépendamment** par l'encodeur, puis agrégés par **Max-Pooling** sur la dimension des chemins pour produire une signature unique de dimension 256 par fonction. Cette approche, inspirée des Deep Sets, confère une invariance à la permutation des chemins et permet une rétropropagation sélective : seuls les chemins ayant fourni le signal maximal reçoivent un gradient.

---

#### 3.5.6 Configuration d'Optimisation

| Hyperparamètre | Valeur |
|---|---|
| Optimiseur | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Taux d'apprentissage | 1 × 10⁻⁴ (constant) |
| Weight decay | 0.01 (défaut AdamW PyTorch) |
| Scheduler LR | Aucun |
| Warmup | Aucun |
| Gradient clipping | Aucun |
| Batch size | 10 séquences |
| Nombre d'époques | 10 |
| Longueur max séquence | 500 tokens (padding ID=0) |
| EMA momentum | m = 0.996, mis à jour à chaque batch |
| Précision mixte (AMP) | Non utilisée (contrainte matérielle) |
| Entraînement distribué | Non (mono-GPU) |

L'absence de scheduler LR est une simplification assumée dans le cadre de cette preuve de concept. Un cosine annealing ou un warmup linéaire serait à envisager pour un entraînement à plus grande échelle.

---

#### 3.5.7 Sauvegarde et Reproductibilité

Un checkpoint est sauvegardé à chaque époque sous la forme :

```
checkpoints/<run_id>/latest.pt
checkpoints/<run_id>/ep<N>.pt
```

Le dictionnaire sauvegardé contient les poids de l'encodeur contextuel, de l'encodeur cible, du prédicteur, ainsi que l'état de l'optimiseur, le numéro d'époque et la loss moyenne — permettant une reprise exacte de l'entraînement. L'identifiant de run encode les hyperparamètres principaux (`YYYYMMDD_HHMMSS_ep{N}_lr{lr}_bs{bs}`), facilitant la traçabilité des expériences.

Les courbes de loss sont enregistrées au format CSV et visualisables via le script `plot_training.py` (courbe brute + lissage EMA α=0.85 + histogramme des deltas par époque).
