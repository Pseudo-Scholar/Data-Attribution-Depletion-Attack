"""
Compatibility entry point.

The complete APS surrogate implementation is in
``Pretrained_Adversarial_Classifier_APS.py``. This file remains so existing
experiment commands that use the historical filename continue to work.
"""

from Pretrained_Adversarial_Classifier_APS import main


if __name__ == "__main__":
    main()

