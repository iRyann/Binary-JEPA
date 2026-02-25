import os

import src.elf_processing_core as elc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_hello_world():
    binary_path = os.path.join(BASE_DIR, "external", "HelloWorld", "hello_world")

    project = elc.lift_binary(binary_path)
    cfg = elc.construct_cfg(project)

    print(f"\n[+] Analyse de : {os.path.basename(binary_path)}")
    print(f"[*] Nombre de nœuds (blocs) dans le CFG : {cfg.graph.number_of_nodes()}")
    print(f"[*] Nombre d'arêtes (sauts) dans le CFG : {cfg.graph.number_of_edges()}")

    # B. Lister les fonctions détectées
    print("\n[+] Fonctions identifiées :")
    for addr, func in project.kb.functions.items():
        # On filtre pour ne pas afficher toutes les fonctions de bibliothèques (externes)
        if not func.is_simprocedure:
            print(f"  - {func.name} à l'adresse {hex(addr)}")

    # C. Focus sur la fonction 'main' (le cœur du programme)
    main_func = project.kb.functions.get("main")
    if main_func:
        print(f"\n[+] Focus sur 'main' ({hex(main_func.addr)}) :")
        print(f"  - Nombre de blocs de base : {len(main_func.block_addrs_set)}")

        # Affichage de l'assembleur du premier bloc du main
        start_block = project.factory.block(main_func.addr)
        print("\n[+] Code assembleur du premier bloc de main :")
        start_block.pp()  # Pretty Print de l'assembleur


if __name__ == "__main__":
    test_hello_world()
