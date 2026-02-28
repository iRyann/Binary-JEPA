# JEPA

## I-JEPA

> [!quote] Résumé en substance
> L'architecture I-JEPA s'abtrait de données manuellement augmentée pour apprendre la sémantique des images.
> Il s'agit :
>
> - d'une approche non générative,
> - d'un apprentissage auto-supervisée
>
> Pour guider cet apprentissage voué à développer la capacité du modèle à prédire des blocs tibles sur la base d'un bloc de contexte, on compte sur une stratégie de masquage.
>
> Passage à l'échelle permis par l'entraînement de _ViT-Huge/14_ sur _ImageNet_ à l'aide 16 GPUs A100 en moins de 72 heures.[^1]

L'espace de prédiction d'I-JEPA se distingue par son niveau d'abstraction plus élevé que celui d'autres architectures, comme par exemple celles des méthodes génératives qui travaillent au niveau du pixel/token.
Les auteurs font remarquer l'importance d'une taille sufisamment large concernant les blocs à prédire.

- view-invariant pretraining
- un modèle simple avec moins de contraintes indiductive permet l'application du modèle à de nombreuses taches.

> [!quote] Définition
> Self-supervised learning is an approach to representation learning in which a system learns to capture the relationships between its inputs.

- [ ] data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language : <https://arxiv.org/abs/2202.03555>
- [ ] A Tutorial on Energy-Based Learning : <https://web.stanford.edu/class/cs379c/archive/2012/suggested_reading_list/documents/LeCunetal06.pdf>

L'intention même d'une JEA correspond bien à celle d'un hashing sémantique, quoiqu'on doive considérer le plus significatif des problèmes dans notre contexte, les collisions.

> [!quote]
> nvariance-based pre-
> training can be cast in the framework of EBMs using a
> Joint-Embedding Architecture (JEA), which learns to out-
> put similar embeddings for compatible inputs, x, y, and dis-
> similar embeddings for incompatible inputs
> he main challenge with JEAs is representation collapse,
> wherein the energy landscape is flat (i.e., the encoder pro-
> duces a constant output regardless of the input).

Dans le cadre des JEPA, on discrimine l'espace d'embedding en calculant la perte sur ce dernier.

- Information additionnelle $z$
- pas d'overlapping entre le contexte, et les cibles ; et ce, afin d'assurer des prédictions non-trivialles.
- les cibles sont tirées de l'output de l'_encodeur cible_

[^1]: <https://arxiv.org/pdf/2301.08243>
