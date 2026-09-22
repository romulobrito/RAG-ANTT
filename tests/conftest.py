"""Coloca a raiz do projeto no path quando o pytest roda a pasta tests/."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
raiz_texto = str(RAIZ)
if raiz_texto not in sys.path:
    sys.path.insert(0, raiz_texto)
