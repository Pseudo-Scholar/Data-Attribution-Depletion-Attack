"""
Compatibility entry point.

The complete APS image-generation implementation is in
``Generate_Perturbed_Images_APS.py``. This historical filename delegates to
that implementation so it cannot accidentally run the old ordinary-PGD-only
code.
"""

from Generate_Perturbed_Images_APS import main


if __name__ == "__main__":
    main()

