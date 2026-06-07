"""Explicit entrypoint for the pan-cancer B3 vs B3+DRKG experiment.

This wrapper calls the shared multi-seed cross-validation runner. By default,
that runner trains both B3 (`cv_base`) and B3+DRKG (`cv_plus_drkg`).
"""

from run_attentivefp_multiseed_cv import main


if __name__ == "__main__":
    main()
