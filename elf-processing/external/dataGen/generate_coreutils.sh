#!/bin/bash
# generate_coreutils.sh
# Génère un dataset SOTA de clones sémantiques basés sur GNU Coreutils.

DATASET_DIR="dataset_coreutils"
SRC_DIR="src_tmp"
COREUTILS_VER="9.10"

mkdir -p "$DATASET_DIR"
mkdir -p "$SRC_DIR"

echo "[*] Installation des dépendances..."
sudo pacman -S gcc clang wget xz-utils build-essential

echo "[*] Téléchargement de GNU Coreutils v${COREUTILS_VER}..."
cd "$SRC_DIR"
wget -qO coreutils.tar.xz "https://github.com/coreutils/coreutils/releases/download/v${COREUTILS_VER}/coreutils-${COREUTILS_VER}.tar.xz"
tar -xf coreutils.tar.xz
cd "coreutils-${COREUTILS_VER}"

# Matrices de compilation
COMPILERS=("gcc" "clang")
OPTIMIZATIONS=("-O0" "-O1" "-O2" "-O3" "-Os")

# On sélectionne les utilitaires les plus iconiques pour ne pas saturer le disque
# (Coreutils contient plus de 100 programmes, on en prend 10 très différents)
TARGET_BINS=("ls" "cat" "cp" "mv" "rm" "sort" "echo" "chmod" "md5sum" "base64")

echo "[*] Début de la matrice de compilation croisée (Cela peut prendre quelques minutes)..."

for comp in "${COMPILERS[@]}"; do
  for opt in "${OPTIMIZATIONS[@]}"; do

    echo "  -> Configuration et compilation avec CC=$comp CFLAGS=$opt..."

    # Nettoyage de la compilation précédente
    make clean >/dev/null 2>&1

    # Configuration avec le compilateur et l'opti choisis
    # --disable-nls accélère la compilation en désactivant les traductions de langue
    ./configure CC="$comp" CFLAGS="$opt" --disable-nls --quiet >/dev/null

    # Compilation multi-coeurs
    make -j$(nproc) --quiet >/dev/null

    # Récupération des binaires cibles
    for bin in "${TARGET_BINS[@]}"; do
      if [ -f "src/$bin" ]; then
        # Formatage du nom : ex: ls_gcc_O3.elf
        OUT_NAME="${bin}_${comp}_${opt#-}.elf"
        cp "src/$bin" "../../$DATASET_DIR/$OUT_NAME"

        # Suppression des symboles de debug pour simuler un binaire "release/malware"
        strip --strip-all "../../$DATASET_DIR/$OUT_NAME" 2>/dev/null
      fi
    done

  done
done

echo "[+] Nettoyage des sources..."
cd ../../
rm -rf "$SRC_DIR"

echo "[+] Terminé"
ls -lh $DATASET_DIR | head -n 15
