#!/usr/bin/env python3
"""Run saturation mutagenesis with a 501-bp score window centered at each variant."""

from __future__ import annotations

import sys

from run_saturation_mutagenesis import main


if __name__ == "__main__":
    if "--score-center" in sys.argv or "--variant-score-width-bp" in sys.argv:
        raise SystemExit("This runner fixes --score-center=variant and width=501 bp")
    sys.argv.extend(("--score-center", "variant", "--variant-score-width-bp", "501"))
    main()
