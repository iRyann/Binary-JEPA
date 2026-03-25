"""
token_colors.py
===============
Catégorisation sémantique des tokens VEX IR et mapping vers des couleurs.

Chaque token appartient à une TokenCategory qui détermine sa couleur
d'affichage dans PathsPanel et IdsPanel. Cette catégorisation est
indépendante du rendu et peut être réutilisée pour des analyses
quantitatives (entropie par catégorie, ratio sémantique, etc.).

Hiérarchie des catégories :
    Token
    ├── REG      VEX_REG_READ, VEX_REG_WRITE
    ├── ARITH    VEX_OP_*  (arithmétique, logique, comparaison)
    ├── MEM      VEX_LOAD, VEX_STORE
    ├── CTRL     VEX_EXIT_COND
    ├── FLOW     JK_* (jump kinds : BORING, CALL, RET, ...)
    ├── TEMP     VEX_WrTmp, VEX_CONST, VEX_OP_*T* (types intermédiaires)
    ├── API      <API_*>
    ├── SYSCALL  <SYSCALL_*>
    └── SPECIAL  <PAD>, <UNK>, <MASK>, <UNLIFTABLE>, VEX_Ist_*
"""

from __future__ import annotations

import colorsys
import hashlib
import re
from enum import Enum

from .theme import TOKEN_CAT_COLORS, TOKEN_CAT_DEFAULT


# ══════════════════════════════════════════════════════════════════════════════
# CATÉGORIES
# ══════════════════════════════════════════════════════════════════════════════

class TokenCategory(str, Enum):
    REG     = "REG"
    ARITH   = "ARITH"
    MEM     = "MEM"
    CTRL    = "CTRL"
    FLOW    = "FLOW"
    TEMP    = "TEMP"
    API     = "API"
    SYSCALL = "SYSCALL"
    SPECIAL = "SPECIAL"


# ══════════════════════════════════════════════════════════════════════════════
# RÈGLES DE CATÉGORISATION (ordre d'évaluation : exact → préfixe → regex)
# ══════════════════════════════════════════════════════════════════════════════

# Correspondances exactes (tokens fréquents, priorité maximale)
_EXACT: dict[str, TokenCategory] = {
    "VEX_REG_READ":   TokenCategory.REG,
    "VEX_REG_WRITE":  TokenCategory.REG,
    "VEX_LOAD":       TokenCategory.MEM,
    "VEX_STORE":      TokenCategory.MEM,
    "VEX_EXIT_COND":  TokenCategory.CTRL,
    "VEX_WrTmp":      TokenCategory.TEMP,
    "VEX_CONST":      TokenCategory.TEMP,
    "JK_BORING":      TokenCategory.FLOW,
    "JK_CALL":        TokenCategory.FLOW,
    "JK_RET":         TokenCategory.FLOW,
    "JK_NODECODE":    TokenCategory.FLOW,
    "JK_SIGTRAP":     TokenCategory.FLOW,
    "<PAD>":          TokenCategory.SPECIAL,
    "<UNK>":          TokenCategory.SPECIAL,
    "<MASK>":         TokenCategory.SPECIAL,
    "<BOS>":          TokenCategory.SPECIAL,
    "<EOS>":          TokenCategory.SPECIAL,
    "<UNLIFTABLE>":   TokenCategory.SPECIAL,
    "<API_UNRESOLVABLECALLTARGET>": TokenCategory.API,
    "<API_UNRESOLVABLEJUMPTARGET>": TokenCategory.API,
}

# Règles préfixe (ordre décroissant de spécificité)
_PREFIX_RULES: list[tuple[str, TokenCategory]] = [
    ("VEX_REG_",    TokenCategory.REG),
    ("VEX_OP_",     TokenCategory.ARITH),
    ("VEX_LOAD",    TokenCategory.MEM),
    ("VEX_STORE",   TokenCategory.MEM),
    ("VEX_EXIT",    TokenCategory.CTRL),
    ("JK_",         TokenCategory.FLOW),
    ("<API_",       TokenCategory.API),
    ("<SYSCALL_",   TokenCategory.SYSCALL),
    ("VEX_Ist_",    TokenCategory.SPECIAL),   # AbiHint, etc.
    ("VEX_",        TokenCategory.TEMP),       # catch-all VEX résiduel
    ("<",           TokenCategory.SPECIAL),    # catch-all tokens spéciaux
]

# Cache LRU manuel (dict simple, pas de limite : le vocab est borné à ~400 tokens)
_cache: dict[str, TokenCategory] = {}


def categorize(token: str) -> TokenCategory:
    """
    Retourne la TokenCategory d'un token.

    Ordre de résolution :
      1. Cache (appels répétés très fréquents en rendu)
      2. Correspondance exacte
      3. Règles préfixe (ordre décroissant de spécificité)
      4. Fallback → SPECIAL
    """
    if token in _cache:
        return _cache[token]

    cat: TokenCategory

    if token in _EXACT:
        cat = _EXACT[token]
    else:
        cat = TokenCategory.SPECIAL  # fallback
        for prefix, candidate in _PREFIX_RULES:
            if token.startswith(prefix):
                cat = candidate
                break

    _cache[token] = cat
    return cat


# ══════════════════════════════════════════════════════════════════════════════
# COULEURS PAR TOKEN INDIVIDUEL
# Chaque token reçoit sa propre couleur (pas seulement sa catégorie) pour
# distinguer visuellement REG_READ de REG_WRITE, JK_CALL de JK_RET, etc.
# ══════════════════════════════════════════════════════════════════════════════

# Tokens sémantiquement importants : couleurs choisies à la main
_TOKEN_COLORS: dict[str, str] = {
    # REG : lecture vs écriture clairement distincts
    "VEX_REG_READ":   "#79c0ff",   # bleu clair  — lecture registre
    "VEX_REG_WRITE":  "#1f6feb",   # bleu foncé  — écriture registre
    # MEM
    "VEX_LOAD":       "#56d4f5",   # cyan vif    — lecture mémoire
    "VEX_STORE":      "#0d7a8a",   # teal foncé  — écriture mémoire
    # CTRL
    "VEX_EXIT_COND":  "#ffa657",   # orange      — branchement conditionnel
    # FLOW : très distincts — sémantique critique
    "JK_BORING":      "#6a5c00",   # ocre sombre — flux séquentiel
    "JK_CALL":        "#388bfd",   # bleu vif    — appel de fonction
    "JK_RET":         "#da3633",   # rouge       — retour
    "JK_NODECODE":    "#3d444d",   # gris foncé  — non décodable
    "JK_SIGTRAP":     "#cf222e",   # rouge vif   — signal trap
    # TEMP
    "VEX_WrTmp":      "#6e7681",   # gris moyen  — temporaire VEX
    "VEX_CONST":      "#b1bac4",   # gris clair  — constante
    # SPECIAL
    "<PAD>":          "#161b22",   # quasi-invisible
    "<UNK>":          "#6e7681",
    "<MASK>":         "#6e7681",
    "<UNLIFTABLE>":   "#3d444d",
    "<API_UNRESOLVABLECALLTARGET>": "#9e4da0",
    "<API_UNRESOLVABLEJUMPTARGET>": "#7a3580",
}

# Teinte de base par catégorie (HSL, H∈[0,1])
_BASE_HUES: dict[str, float] = {
    "REG":     0.595,
    "ARITH":   0.330,
    "MEM":     0.510,
    "CTRL":    0.075,
    "FLOW":    0.150,
    "TEMP":    0.570,
    "API":     0.750,
    "SYSCALL": 0.020,
    "SPECIAL": 0.000,
}


def _category_token_color(token: str, cat: "TokenCategory") -> str:
    """
    Génère une couleur unique et déterministe pour un token dans la famille
    de teinte de sa catégorie. Utilise le hash MD5 du nom pour la variation.
    """
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    base_hue = _BASE_HUES.get(cat.value, 0.5)

    # Variation de teinte ±0.04 → reste reconnaissable comme la même famille
    hue_var = ((h >> 16) % 9 - 4) * 0.01
    hue = (base_hue + hue_var) % 1.0

    # Lightness 0.42–0.67 (6 niveaux) et saturation 0.60–0.90 (7 niveaux)
    lightness   = 0.42 + ((h >> 8) % 6) * 0.05
    saturation  = 0.60 + (h       % 7) * 0.05

    if cat.value == "SPECIAL":
        lightness  = 0.13 + (h % 4) * 0.04
        saturation = 0.10
    elif cat.value == "TEMP":
        saturation = 0.28 + (h % 5) * 0.08
        lightness  = 0.38 + (h % 6) * 0.05

    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# Cache des couleurs par token (calcul unique par token vu)
_color_cache: dict[str, str] = {}


def hex_color(token: str) -> str:
    """Couleur hexadécimale unique par token (pas seulement par catégorie)."""
    if token in _color_cache:
        return _color_cache[token]
    if token in _TOKEN_COLORS:
        color = _TOKEN_COLORS[token]
    else:
        color = _category_token_color(token, categorize(token))
    _color_cache[token] = color
    return color


# ══════════════════════════════════════════════════════════════════════════════
# ABRÉVIATIONS D'AFFICHAGE
# Noms courts pour les cellules compactes (≤ 6 caractères cibles).
# ══════════════════════════════════════════════════════════════════════════════

_ABBREV_EXACT: dict[str, str] = {
    "VEX_REG_READ":   "REG_R",
    "VEX_REG_WRITE":  "REG_W",
    "VEX_LOAD":       "LOAD",
    "VEX_STORE":      "STOR",
    "VEX_EXIT_COND":  "EXIT",
    "VEX_WrTmp":      "TMP",
    "VEX_CONST":      "CST",
    "JK_BORING":      "BORE",
    "JK_CALL":        "CALL",
    "JK_RET":         "RET",
    "JK_NODECODE":    "NDEC",
    "JK_SIGTRAP":     "TRAP",
    "<PAD>":          "PAD",
    "<UNK>":          "UNK",
    "<MASK>":         "MASK",
    "<BOS>":          "BOS",
    "<EOS>":          "EOS",
    "<UNLIFTABLE>":   "UNLT",
    "<API_UNRESOLVABLECALLTARGET>": "UNRC",
    "<API_UNRESOLVABLEJUMPTARGET>": "UNRJ",
}

# Règles de transformation pour les cas génériques
_OP_RE   = re.compile(r"^VEX_OP_(.+)$")
_API_RE  = re.compile(r"^<API_(.+)>$")
_SYSC_RE = re.compile(r"^<SYSCALL_(.+)>$")
_JK_RE   = re.compile(r"^JK_(.+)$")
_IST_RE  = re.compile(r"^VEX_Ist_(.+)$")


def display_name(token: str) -> str:
    """
    Nom d'affichage court d'un token (≤ 7 caractères).

    Exemples :
        VEX_OP_ADD  → ADD
        VEX_OP_64T  → 64T
        <API_MALLOC>  → MALC
        <SYSCALL_READ> → READ
        JK_BORING   → BORE
    """
    if token in _ABBREV_EXACT:
        return _ABBREV_EXACT[token]

    m = _OP_RE.match(token)
    if m:
        return m.group(1)[:6]

    m = _API_RE.match(token)
    if m:
        return m.group(1)[:6]

    m = _SYSC_RE.match(token)
    if m:
        return m.group(1)[:6]

    m = _JK_RE.match(token)
    if m:
        return m.group(1)[:6]

    m = _IST_RE.match(token)
    if m:
        return m.group(1)[:6]

    # fallback brut tronqué
    return token[:7]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRE : légende des catégories (pour PathsPanel)
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_LEGEND: list[tuple[str, str]] = [
    (cat.value, TOKEN_CAT_COLORS.get(cat.value, TOKEN_CAT_DEFAULT))
    for cat in TokenCategory
]
