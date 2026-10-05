# Data Attribution Depletion Attack-Python
This code is the implementation of the paper **Gain-Stealing and Loss-Amplifying: Universal Data Attribution Depletion in Machine Learning with In-Run Data Shapley**.

## An Implementation of Data Attribution Depletion Attack in Python

Data Attribution Depletion Attack is a novel and potent attack that successfully manipulates the In-Run Data Shapley mechanism, significantly **increasing malicious clients' gains and degrading model performance**.

Additionally, it **exhibits superior generality and real-world threat**, particularly with backdoor poisoning and adversarial perturbation strategies designed for both black-box and white-box threat models.

## Requirements

- Requires at least **Python 3.12.0**.

## Examples
### Backdoor Poisoning Strategy

- The **Pixel-Pattern Injection Strategy** is a backdoor poisoning technique specifically designed for handwritten digit recognition tasks. This method generates poisoned samples by injecting a set of bright pixel patterns into the lower-right corner of benign inputs. For these poisoned instances, the attacker modifies the label of digit i to (i+1)(mod10).
- To make the poisoned inputs appear benign, the trigger must be as inconspicuous as possible. First, the strategy **reduces the trigger's visibility** by subtly adjusting original pixel values instead of directly replacing them. Second, to **ensure the trigger remains visible despite common data augmentation techniques**, such as random crops and flips, the pattern is replicated across all four corners of the image. This dual approach significantly enhances both the stealthiness and robustness of the attack.
- The **MNIST dataset** used by this strategy will be automatically loaded via the PyTorch library.
- The code implementing this strategy is located at `code/Backdoor_Poisoning_Strategy.py`.
- The poisoned samples for this strategy are located at `data/poisoned_sample/Backdoor_Poisoning_Strategy`.


### Adversarial Perturbation Strategy
- The **Adversarial Perturbation Strategy** aims to generate effective and label-consistent poisoned samples. This strategy applies adversarial perturbations to benign samples, making them harder to classify, thereby creating the final poisoned samples after injecting triggers. 
- Specifically, it maximizes the loss of an independently trained model on the poisoned input while maintaining a bounded **lp norm** distance from the original input. Additionally, it incorporates a **locally Shapley value-guided gradient alignment constraint** to ensure that the poisoned input contributes positively to model training, thereby enhancing the success rate of the attack. Ultimately, the generated samples are **visually imperceptible**, and the model may mistakenly **overestimate their contribution** to training, significantly increasing the attack's stealthiness and robustness.
- The **CIFAR10 dataset** used by this strategy will be automatically loaded via the PyTorch library.
- The code implementing this strategy is located at `code/Adversarial_Perturbation_Strategy.py`.
- The poisoned samples for this strategy are located at `data/poisoned_sample/Adversarial_Perturbation_Strategy`.




