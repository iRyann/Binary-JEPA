# L'émergence latente : au-delà de l'embedding explicite

Contrairement à la majorité des travaux de l'état de l'art qui traitent l'embedding comme un objectif premier et explicite — souvent contraint par des tâches de reconstruction syntaxique (Masked Language Modeling) ou d'isomorphisme de graphes —, notre approche opère un renversement de paradigme.

En nous appuyant sur l'architecture JEPA (Joint-Embedding Predictive Architecture), nous ne forçons pas la création d'un espace vectoriel par mimétisme syntaxique. L'architecture apprend à prédire la représentation latente d'un bloc de code manquant en se basant sur son contexte. Dès lors, pour minimiser son erreur de prédiction, le modèle est mathématiquement contraint d'ignorer le bruit stochastique (les variations d'obfuscation, le code mort) pour ne capter que l'essence comportementale du programme.

L'embedding sémantique n'est plus l'artefact direct d'une fonction de hachage ; il jaillit de façon sous-jacente comme la seule structure topologique capable de lier le contexte à son évolution. L'intention première du programme se retrouve ainsi cristallisée, non pas parce qu'on a cherché à l'encoder, mais parce qu'elle était la condition sine qua non pour la prédire.
