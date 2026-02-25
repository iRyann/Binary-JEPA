import os

import angr

"""Ce module fournit des fonctions utilitaires pour analyser un binaire avec angr, 
notamment pour construire et afficher le CFG (Control Flow Graph) du projet."""


def print_cfg(cfg: angr.analyses.CFGFast, project: angr.Project) -> None:
    """
    Affiche des informations sur le CFG et les fonctions identifiées dans le projet.
    """
    print(f"\n[+] Analyse de : {cfg.project.filename}")
    print(f"[*] Nombre de nœuds (blocs) dans le CFG : {cfg.graph.number_of_nodes()}")
    print(f"[*] Nombre d'arêtes (sauts) dans le CFG : {cfg.graph.number_of_edges()}")

    print("\n[+] Fonctions identifiées :")
    for addr, func in project.kb.functions.items():
        # On filtre pour ne pas afficher toutes les fonctions de bibliothèques (externes)
        if not func.is_simprocedure:
            print(f"  - {func.name} à l'adresse {hex(addr)}")

    main_func = project.kb.functions.get("main")
    if main_func:
        print(f"\n[+] Focus sur 'main' ({hex(main_func.addr)}) :")
        print(f"  - Nombre de blocs de base : {len(main_func.block_addrs_set)}")

        start_block = project.factory.block(main_func.addr)
        print("\n[+] Code assembleur du premier bloc de main :")
        start_block.pp()  # Pretty Print de l'assembleur
