import os

import angr

"""
Ce module contient des fonctions utilitaires pour analyser les CFGs générés par Angr.
Il permet d'afficher des informations sur les fonctions identifiées, les blocs de base, exceptions, etc. L'objectif est de faciliter la compréhension du CFG et d'aider à l'identification de patterns intéressants.:
"""
def print_cfg(cfg: angr.analyses.CFGFast, project: angr.Project) -> None:
    print(f"\n[+] Analyse de : {cfg.project.filename}")
    print(f"[*] Nombre de nœuds (blocs) dans le CFG : {cfg.graph.number_of_nodes()}")
    print(f"[*] Nombre d'arêtes (sauts) dans le CFG : {cfg.graph.number_of_edges()}")

    print("\n[+] Fonctions identifiées :")
    for addr, func in project.kb.functions.items():
        # On filtre pour ne pas afficher toutes les fonctions de bibliothèques (externes)
        if not func.is_simprocedure:
            print(f"  - {func.name} à l'adresse {hex(addr)}")

    main_func = project.kb.functions.get('main')
    if main_func:
        print(f"\n[+] Focus sur 'main' ({hex(main_func.addr)}) :")
        print(f"  - Nombre de blocs de base : {len(main_func.block_addrs_set)}")
        
        start_block = project.factory.block(main_func.addr)
        print("\n[+] Code assembleur du premier bloc de main :")
        start_block.pp() # Pretty Print de l'assembleur
