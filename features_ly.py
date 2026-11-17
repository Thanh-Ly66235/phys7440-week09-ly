"""Sequence-only substitution descriptors for the PhysChem-Ridge baseline.

These descriptors know only WHICH amino acid is replaced by WHICH; they know
nothing about where the residue sits in the fold. The AlphaFold-informed model
of Weeks 9-13 adds structural context (burial, pLDDT, AlphaMissense) on top of
exactly these columns, so the comparison isolates what the structure adds.

Scales (standard literature tables):
  * Kyte & Doolittle (1982) hydropathy, J. Mol. Biol. 157, 105.
  * Zamyatnin (1972) residue volumes in A^3, Prog. Biophys. Mol. Biol. 24, 107.
  * Pace & Scholtz (1998) helix propensity, kcal/mol relative to Ala,
    Biophys. J. 75, 422.
  * BLOSUM62 (Henikoff & Henikoff 1992), embedded below so that the notebook
    needs no Biopython.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

AA = "ARNDCQEGHILKMFPSTWYV"

HYDROPATHY_KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
                     H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
                     T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOLUME_A3 = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4,
                 G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9,
                 P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHARGE_PH7 = dict(D=-1, E=-1, K=1, R=1)            # His treated as neutral
HELIX_PACE_SCHOLTZ = dict(A=0.00, L=0.21, R=0.21, M=0.24, K=0.26, Q=0.39, E=0.40,
                          I=0.41, W=0.49, S=0.50, Y=0.53, F=0.54, H=0.61, V=0.61,
                          N=0.65, T=0.66, C=0.68, D=0.69, G=1.00, P=3.16)

_BLOSUM62_ROWS = """\
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4"""
BLOSUM62 = {}
for _row in _BLOSUM62_ROWS.splitlines():
    _a, *_vals = _row.split()
    for _b, _v in zip(AA, _vals):
        BLOSUM62[(_a, _b)] = int(_v)

FEATURES = ["blosum62", "d_hydropathy", "d_volume_A3", "abs_d_volume_A3",
            "d_charge", "d_helix_kcal", "to_gly", "from_gly", "to_pro", "from_pro"]

# wild type, residue number (+ optional PDB insertion code, e.g. 'L27CN'), mutant
_MUT_RE = re.compile(r"^([A-Z])(-?\d+[A-Z]?)([A-Z])$")


def parse_mutation(m: str) -> tuple[str, str, str]:
    """'S11A' -> ('S', '11', 'A');  'L27CN' (insertion code C) -> ('L', '27C', 'N')."""
    hit = _MUT_RE.match(m.strip())
    if not hit or hit.group(1) not in AA or hit.group(3) not in AA:
        raise ValueError(f"not a single-point mutation: {m!r}")
    return hit.group(1), hit.group(2), hit.group(3)


def substitution_features(wt: str, mut: str) -> dict[str, float]:
    """Context-free descriptors of the substitution wt -> mut (mut minus wt)."""
    return {
        "blosum62": BLOSUM62[(wt, mut)],
        "d_hydropathy": HYDROPATHY_KD[mut] - HYDROPATHY_KD[wt],
        "d_volume_A3": VOLUME_A3[mut] - VOLUME_A3[wt],
        "abs_d_volume_A3": abs(VOLUME_A3[mut] - VOLUME_A3[wt]),
        "d_charge": CHARGE_PH7.get(mut, 0) - CHARGE_PH7.get(wt, 0),
        "d_helix_kcal": HELIX_PACE_SCHOLTZ[mut] - HELIX_PACE_SCHOLTZ[wt],
        "to_gly": float(mut == "G"), "from_gly": float(wt == "G"),
        "to_pro": float(mut == "P"), "from_pro": float(wt == "P"),
    }


def featurise(mutations: pd.Series) -> pd.DataFrame:
    """Feature matrix (one row per mutation string, columns = FEATURES)."""
    rows = []
    for m in mutations:
        wt, _, mut = parse_mutation(m)
        rows.append(substitution_features(wt, mut))
    return pd.DataFrame(rows, index=mutations.index)[FEATURES]


def reverse_mutation(m: str) -> str:
    """'S11A' -> 'A11S' (used for the antisymmetry check)."""
    wt, pos, mut = parse_mutation(m)
    return f"{mut}{pos}{wt}"
