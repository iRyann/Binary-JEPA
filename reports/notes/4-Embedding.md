# De la syntaxe, à l'embedding, la sémasiologie appliquée à l'informatique

> [!abstract]
> Du programme en tant que tel, comme fichier ELF, à la sémantique instillée par l'auteur de ce dernier, et jusqu'à la représentation qu'il en est donnée dans un espace approprié pour le Machine Learning, la substance manipulée se voit constamment remisée par ces différentes _représentations_.
> Derrière cet agrégat informationnel projeté dans des espaces divers se pose la question de la perte informationnelle -- et donc réciproquement de la conservation de celle-ci --, et de la fidélité à l'intention première matérialisée par le programme.
> Ces interrogations sont évidemment au fondement de l'embedding de code binaire, ou l'apprentissage de représentations vectorielles pour l'analyse binaire, visant à transformer des séquences d'instructions non structurées en vecteurs numériques denses de faible dimension capturant leur sémantique.

### Cartographie de la Représentation Binaire : De la Syntaxe à l'Espace Latent

L'apprentissage de représentations pour l'analyse binaire ambitionne de transmuter un agrégat d'instructions brutes en une projection vectorielle dense. Cette démarche s'inscrit dans une quête de similarité sémantique (BCSD), où l'enjeu fondamental est d'élaguer les scories syntaxiques pour ne conserver que la fidélité à l'intention première du programme. Historiquement, cette abstraction s'est heurtée à plusieurs paradigmes de modélisation.

#### 1. L'illusion lexicale : L'approche séquentielle (Word2Vec et RNN)

Dans ses balbutiements, la recherche a postulé un isomorphisme naïf entre le langage machine et les idiomes naturels. L'instruction y est traitée comme un lexème, et le bloc de base comme une phrase.

- **Les fondations :** Des modèles comme **Asm2Vec** ou **SAFE** s'appuient sur des marches aléatoires (random walks) et des réseaux récurrents pour forger un embedding sans ingénierie manuelle.
- **La perte informationnelle :** Cette assimilation montre rapidement ses limites. La volatilité de la syntaxe assembleur face aux optimisations des compilateurs entraîne une dégradation sémantique majeure. L'intention se perd dans le bruit lexical.

#### 2. L'empreinte topologique : La modélisation structurelle (GNN)

Consciente que l'exécution d'un programme échappe à la stricte linéarité textuelle, la recherche s'est tournée vers la capture de la topologie via les Graphes de Flux de Contrôle (CFG).

- **L'ancrage structurel :** Le modèle **Gemini** fut le pionnier de cette projection, figeant les attributs du CFG dans un réseau de neurones sur graphes (Structure2Vec). Des architectures hybrides (**GMN**) ont ensuite raffiné cette approche par des mécanismes d'attention inter-graphes.
- **La limite opérationnelle :** Bien que fidèles à l'architecture spatiale du programme, ces modèles souffrent d'une lourdeur calculatoire ($O(N^2)$) rédhibitoire pour une application à l'échelle industrielle.

#### 3. L'attention globale : La suprématie des Transformers

Le changement de paradigme actuel repose sur l'architecture Transformer, capable d'appréhender la sémantique complexe par le prisme de l'auto-attention à longue distance, remisant les approches précédentes.

- **L'analyse contextuelle :** Des modèles comme **PalmTree** ou **jTrans** pré-entraînent de vastes réseaux (de type BERT) pour assimiler la dynamique du flux de contrôle. **Trex** pousse la logique plus loin en s'appuyant sur des micro-traces d'exécution, substituant l'analyse statique pure par une sémantique comportementale.
- **Le biais du MLM :** Ces architectures s'entraînent souvent via la prédiction de masques (Masked Language Modeling), une heuristique qui force parfois le modèle à mémoriser des artefacts syntaxiques de bas niveau plutôt qu'à extraire l'essence conceptuelle de l'algorithme.

#### 4. La quête de l'invariant : La résistance à l'Obfuscation

Le code hostile (malwares) cherche délibérément à distordre sa propre représentation pour tromper l'analyse, posant la question de la robustesse de l'espace de projection.

- **L'effondrement topologique :** L'aplatissement du flux de contrôle ou l'insertion de code mort (Junk code) détruisent l'intégrité du CFG.
- **La parade sémantique :** Pour conserver l'intention première, des modèles comme **ORCAS** abandonnent le CFG au profit du graphe de dominance sémantique (DESG), s'affranchissant des altérations superficielles pour traquer la dépendance inaliénable des données.

#### 5. La cristallisation de l'intention : Le Hashing Sémantique Profond

La projection de millions de binaires pose inévitablement la question de la concision de l'information. L'état de l'art s'éloigne de l'embedding flottant pour embrasser le hachage sémantique.

- **La compression ultime :** Il s'agit de contraindre le réseau de neurones à projeter l'agrégat informationnel dans un espace de Hamming. Des percées comme **KEENHash** parviennent à condenser la sémantique d'un programme entier en une signature compacte, sacrifiant la résolution microscopique pour garantir un passage à l'échelle industriel et une comparaison algorithmique foudroyante.

#### 6. Les illusions sémantiques : La robustesse face à l'adversité

Malgré leur sophistication, les espaces latents contemporains demeurent vulnérables aux distorsions calculées.

- **La manipulation de la représentation :** Des attaques telles qu'**AIMA** démontrent qu'il est possible de déplacer stratégiquement des segments d'instructions pour préserver l'exécution matérielle tout en altérant drastiquement la projection mathématique. L'enjeu futur de la sémasiologie informatique sera de garantir l'immuabilité de l'embedding face à ces perturbations ciblées.
