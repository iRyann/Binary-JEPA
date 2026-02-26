import json
import sys
from pathlib import Path


def generate_dot_for_function(jsonl_file: str, target_func: str):
    paths = []

    # 1. Extraction des chemins pour la fonction ciblée
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("func_addr") == target_func:
                paths.append(data["tokens"])

    if not paths:
        print(f"❌ Aucune donnée trouvée pour la fonction {target_func}.")
        return

    print(
        f"[*] {len(paths)} chemins trouvés pour {target_func}. Génération du graphe..."
    )

    # 2. Construction de l'arbre (Syntaxe Graphviz)
    dot_code = [
        "digraph BagOfPaths {",
        "  rankdir=TB;",  # Top to Bottom
        '  node [shape=box, style="rounded,filled", fillcolor="#e0eaf5", fontname="Courier", fontsize=10];',
        '  edge [color="#555555"];',
        f'  root [label="ENTRY\\n{target_func}", shape=ellipse, fillcolor="#ffcc00"];',
    ]

    node_id_counter = 1
    # Dictionnaire pour fusionner les préfixes communs (L'arbre)
    trie_root = {"id": "root", "children": {}}

    for path in paths:
        current_node = trie_root
        for token in path:
            # Si le token diverge, on crée une nouvelle branche
            if token not in current_node["children"]:
                child_id = f"n_{node_id_counter}"
                node_id_counter += 1

                # Couleurs spéciales pour les API et les terminaux
                fillcolor = "#e0eaf5"
                if token.startswith("<API_"):
                    fillcolor = "#ffcccc"  # Rouge clair pour les API
                elif token.startswith("JK_"):
                    fillcolor = "#ccffcc"  # Vert clair pour les fins de chemins

                dot_code.append(
                    f'  {child_id} [label="{token}", fillcolor="{fillcolor}"];'
                )
                dot_code.append(f'  {current_node["id"]} -> {child_id};')

                current_node["children"][token] = {"id": child_id, "children": {}}

            # On avance dans l'arbre
            current_node = current_node["children"][token]

    dot_code.append("}")

    # 3. Écriture du fichier
    out_name = f"graph_{target_func}.dot"
    with open(out_name, "w") as f:
        f.write("\n".join(dot_code))

    print(f"✅ Graphe généré : {out_name}")
    print(
        "👉 Allez sur https://dreampuf.github.io/GraphvizOnline/ et collez le contenu du fichier !"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_paths.py <fichier.jsonl> <func_addr>")
        sys.exit(1)

    generate_dot_for_function(sys.argv[1], sys.argv[2])
