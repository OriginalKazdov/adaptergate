"""Bundled CPU-only demos for adaptergate.

Three demos, all runnable in seconds on any laptop:

  - ``classifier`` : aggregate regression via sklearn TF-IDF + LR classifier.
  - ``silent``     : silent slice regression — the case adaptergate exists for.
                    Same data, gate run twice (with and without ``--slice-epsilon``)
                    to show the safety-net effect.
  - ``sql``        : generative scorer — adapters emit SQL strings, scorer does
                    AST-equality (or normalized string equality).

Invoked via ``adaptergate demo {classifier|silent|sql}``.
"""

from adaptergate.demos.classifier import run as run_classifier
from adaptergate.demos.silent import run as run_silent
from adaptergate.demos.sql import run as run_sql

__all__ = ["run_classifier", "run_silent", "run_sql"]
