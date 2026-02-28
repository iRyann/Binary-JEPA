# Semantic-Invariant Hashing for Executable Binaries via Deep Learning Models

## Introduction

Face à la recrudescence des malwares polymorphes, les signatures cryptographiques classiques (MD5, SHA256) sont obsolètes : un seul bit modifié invalide la détection.
Notre projet vise à développer une approche de "hachage sémantique" appliquée aux binaires.
L'objectif est de produire une signature stable identique pour un programme A et un programme A' (modifié/obfusqué) qui effectuent des traitements identiques ou similaires.
Le cœur du projet résidera dans l'exploration et la sélection d'une architecture de réseau de neurones adaptée au traitement de séquences d'instructions (Bytecode/Opcodes).

## Objectif du projet

Concevoir et évaluer un prototype de fonction de hachage capable de calculer un hash identique entre deux exécutables dont le contenu binaire brut est différent mais qui sont sémantiquement égaux.
Le projet devra déterminer quelle architecture de réseau de neurones offre le meilleur compromis performance/précision pour traiter un programme informatique.

## Résultats Attendus

Une étude comparative : Justification du choix de l'architecture neuronale pour le traitement de binaires.
Un prototype fonctionnel de hachage de fichiers exécutables elf64, invariant à la sémantique ou à défaut, à des altérations mineures.
