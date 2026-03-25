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


def hex_color(token: str) -> str:
    """Couleur hexadécimale (#rrggbb) associée à un token."""
    return TOKEN_CAT_COLORS.get(categorize(token).value, TOKEN_CAT_DEFAULT)


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
