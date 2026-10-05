"""
Explicit APS IRDS entry point.

The implementation lives in ``Adversarial_Perturbation_Strategy_APS.py``.
This named entry point makes the clean/APS victim-side IRDS stage explicit
and, by default, produces only:

    shapley_original_APS.csv
    shapley_aps.csv
"""

from Adversarial_Perturbation_Strategy_APS import main


if __name__ == "__main__":
    main()

