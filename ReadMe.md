# Data Attribution Depletion Attack

Data Attribution Depletion Attack is a security evaluation and reproduction codebase for In-Run Data Shapley (IRDS)-based data attribution. It accompanies the paper *Gain-Stealing and Loss-Amplifying: Universal Data Attribution Depletion in Machine Learning with In-Run Data Shapley*. The repository contains multiple instantiated attack strategies, IRDS training pipelines, attribution-value export utilities, evaluation metrics, and Neural Cleanse-based mitigation experiments.

## Requirements

- Python 3.10 or a compatible version
- PyTorch
- TorchVision
- NumPy
- Pillow
- Matplotlib
- tqdm

Some experiments require GPU acceleration, particularly APS training on CIFAR-10, ViT-B/16 training on ImageNet-100, and Neural Cleanse trigger reverse engineering. Datasets, pretrained model checkpoints, and other large files may need to be prepared or extracted separately for each experiment.

We recommend running the experiments in an isolated Conda or virtual environment and installing PyTorch and TorchVision versions compatible with the local CUDA installation. GPU acceleration is generally not required when running only the metric calculation scripts.

## Project Overview

Data attribution methods measure the contribution of training data to model performance and are important in collaborative learning, data marketplaces, and reward allocation. Classical Data Shapley requires repeated retraining on different data subsets and is therefore expensive at scale. In-Run Data Shapley addresses this limitation by estimating sample contributions during training through the alignment between training-sample gradients and validation gradients. The resulting attribution values can be used for reward allocation and data cleaning.

This project studies the security risks of such efficient attribution mechanisms. An attacker injects carefully crafted samples so that IRDS produces systematic attribution bias for malicious or manipulated samples. The attack has two main objectives:

- **Gain-Stealing:** cause poisoned samples to receive inflated positive contributions and enable a malicious client to obtain disproportionate rewards.
- **Loss-Amplifying:** disrupt attribution-based data cleaning so that genuinely beneficial samples are incorrectly removed or harmful samples are retained, increasing the loss of subsequent retraining.

The codebase follows the DADA attack framework described in the paper. It provides black-box backdoor poisoning strategies, a white-box adversarial perturbation strategy, and Neural Cleanse-based detection and filtering pipelines for evaluating possible mitigations.

## Design Motivation

The main advantage of IRDS is scalability: instead of repeatedly retraining models on many subsets, it estimates data contributions from gradient information collected during training. This efficiency also creates a potential attack surface. If an attacker can influence the local loss or gradient direction of selected training samples, the attacker may change the training-validation gradient inner product and consequently manipulate the accumulated global attribution value.

The project is designed to study three questions:

1. How can conventional poisoning, backdoor, or adversarial techniques be adapted to manipulate data attribution values rather than only inference-time predictions?
2. How general is the attack across data modalities and model architectures, including handwritten digits, face images, natural images, and high-resolution ImageNet subsets?
3. How do these attacks affect reward allocation, attribution-based data cleaning, and attribution stability for benign clients, and to what extent can classical backdoor defenses mitigate them?

## Method Framework

The experiments in this repository follow the stages below:

1. Generate paired clean and poisoned samples. Each strategy uses a modality-specific construction method, such as pixel-pattern injection, accessory-pattern injection, adversarial perturbation, or hidden-trigger optimization.
2. Run IRDS training separately with the clean data and with the poisoned data.
3. Accumulate first-order In-Run Data Shapley estimates during training and export two aligned CSV files.
4. Compute ASR, GSR, GVR, and LAR from the clean and poisoned attribution values and the corresponding training loss curves.
5. For PPIS, APIS, APS, and HTBA-based attacks, optionally run Neural Cleanse detection and filtering experiments to compare attack effectiveness before and after mitigation.

The main attack instances included in the repository are:

- **PPIS:** Pixel-Pattern Injection Strategy for handwritten digit recognition on MNIST.
- **APIS:** Accessory-Pattern Injection Strategy for face recognition on YouTube Aligned Face.
- **APS:** Adversarial Perturbation Strategy for label-consistent adversarial poisoning on CIFAR-10.
- **HTBA-based:** Hidden Trigger Backdoor Attack-based strategy for an ImageNet-100 subset with a ViT-B/16 model.

## Repository Structure

```text
experiment_package/
├── README.md
├── README_CN.md
└── Data Attribution Depletion Attack/
    ├── code/
    │   ├── Backdoor_Poisoning_Strategy_PPIS.py
    │   ├── Backdoor_Poisoning_Strategy_PPIS_IRDS.py
    │   ├── Backdoor_Poisoning_Strategy_APIS.py
    │   ├── Backdoor_Poisoning_Strategy_APIS_IRDS.py
    │   ├── Adversarial_Perturbation_Strategy_APS.py
    │   ├── Generate_Perturbed_Images_APS.py
    │   ├── Pretrained_Adversarial_Classifier_APS.py
    │   ├── Hidden_Trigger_Backdoor_Strategy_HTBA.py
    │   ├── Hidden_Trigger_Backdoor_Strategy_HTBA_IRDS.py
    │   ├── Neural_Cleanse_PPIS*.py
    │   ├── Neural_Cleanse_APIS*.py
    │   ├── Neural_Cleanse_APS*.py
    │   ├── Neural_Cleanse_HTBA*.py
    │   ├── Attack_Success_Rate.py
    │   ├── Gain_Stealing_Rate.py
    │   ├── Gain_Volatility_Rate.py
    │   ├── Loss_Amplifying_Rate.py
    │   └── Data_Attribution_Metrics_Common.py
    └── data/
        ├── MNIST/
        ├── CIFAR10/
        ├── YouTube Aligned Face/
        └── ImageNet/
```

`README.md` is located at the root of the experiment package, while `README_CN.md` contains the Chinese version. The `Data Attribution Depletion Attack/code` directory contains poisoning-data generation, IRDS training, metric calculation, and mitigation scripts. The `Data Attribution Depletion Attack/data` directory is used for public datasets, accessory materials, pretrained checkpoints, and ImageNet subsets. Experiments generate poisoned samples, mapping files, training curves, Neural Cleanse reports, and attribution-value CSV files in the specified output directories.

## Experiments

The repository covers the following experiment settings:

- **MNIST + PPIS:** Evaluate the effect of pixel-pattern triggers on IRDS attribution values.
- **YouTube Aligned Face + APIS:** Evaluate accessory-pattern injection in face recognition and data attribution.
- **CIFAR-10 + APS:** Generate label-consistent poisoned samples using adversarial perturbations and a Shapley-guided gradient-alignment constraint.
- **ImageNet-100 + HTBA-based:** Evaluate hidden-trigger poisoning on high-resolution natural images using a ViT-B/16 architecture.
- **Neural Cleanse mitigation:** Apply trigger reverse engineering, anomalous target-label detection, and suspicious-sample filtering to the supported poisoning strategies.
- **Metric evaluation:** Compute attack success, reward manipulation, normal-client reward volatility, and retraining-loss amplification from paired IRDS outputs.

For each strategy, the principal outputs are two aligned sample-attribution CSV files: one from baseline training without poisoned samples and one from training with poisoned or mitigated samples. The metric scripts use these paired attribution files for downstream analysis.

## Datasets

The experiments use the following datasets and auxiliary resources:

- **MNIST:** Used by the PPIS pixel-pattern injection experiment.
- **CIFAR-10:** Used by the APS adversarial perturbation and pretrained-classifier experiments.
- **YouTube Aligned Face:** Used by the APIS accessory-pattern injection experiment. The aligned face images and accessory materials are required.
- **ImageNet-100:** Used by the HTBA-based high-resolution natural-image experiment. The data should follow an ImageNet-style layout:

  ```text
  train/<class_name>/<image>
  val/<class_name>/<image>
  ```

The repository currently includes MNIST and CIFAR-10 data files, as well as a compressed YouTube Aligned Face archive and accessory images. Before running the full experiments, verify that YouTube Aligned Face and ImageNet-100 have been extracted into the directory structures expected by the corresponding scripts.

## Configuration

The experiment entry points are primarily configured through command-line arguments. Common parameters include:

- `--poison-ratio`: Poisoning ratio.
- `--epochs`: Number of IRDS or model-training epochs.
- `--batch-size`: Training batch size.
- `--learning-rate`: Learning rate.
- `--validation-size`: Number of validation samples used for IRDS gradient alignment.
- `--seed`: Random seed.
- `--output-dir` or `--output-root`: Output directory.

APIS additionally supports accessory-mixing parameters such as `--alpha` and `--accessory-scales`. APS supports perturbation parameters such as `--epsilon`, `--norm`, and `--eta`. HTBA-based experiments support ImageNet/ViT parameters such as `--source-class`, `--target-class`, `--trigger-size`, `--integration-mode`, `--trainable-scope`, and `--irds-parameter-scope`.

The Neural Cleanse mitigation scripts generally expose parameters such as `--nc-steps`, `--nc-batch-size`, `--mad-threshold`, `--filter-threshold`, and `--mitigation-mode`. For low-cost debugging, reduce the number of epochs, scanned labels, training samples, or Neural Cleanse optimization steps. For final experiments, use the dataset scale and training settings specified by the target evaluation.

## Evaluation

The project uses four primary metrics to evaluate data attribution depletion attacks:

- **ASR (Attack Success Rate):** Compares the IRDS values of poisoned samples with those of their clean counterparts and counts pairs satisfying either the Gain-Stealing or Loss-Amplifying condition.
- **GSR (Gain-Stealing Rate):** Measures the reward obtained by a malicious client after uploading poisoned data relative to the reward obtained after uploading clean data.
- **GVR (Gain-Volatility Rate):** Measures the change in the total rewards received by normal clients under the poisoned scenario relative to the clean scenario.
- **LAR (Loss-Amplifying Rate):** Measures the increase in retraining loss caused by attribution-based data cleaning errors after samples with positive Shapley values are removed.

The paper shows that DADA can systematically manipulate the IRDS values of selected poisoned samples while keeping the attribution of most benign samples relatively stable. Black-box backdoor strategies are easier to deploy but depend on task-specific triggers. APS directly targets the training-validation gradient inner product through a local Shapley-guided gradient-alignment constraint and generally provides stronger attribution manipulation. Neural Cleanse can provide partial mitigation against explicit-trigger black-box attacks, but has limited effectiveness against label-consistent, sample-specific white-box perturbations.

## Usage

The recommended experiment sequence is:

1. Prepare the datasets and auxiliary resources. Verify the expected structures under `data/MNIST`, `data/CIFAR10`, `data/YouTube Aligned Face`, and `data/ImageNet`.
2. Generate poisoned samples and mapping files using the entry point for the selected strategy.
3. Run the corresponding IRDS training script to obtain clean and poisoned `shapley_*.csv` files.
4. Use the metric scripts to compute ASR, GSR, GVR, and LAR.
5. To evaluate mitigation, run the corresponding `Neural_Cleanse_<Strategy>_IRDS.py` script and compare attribution values and metrics before and after defense.

Example entry points:

```powershell
# Run this command from the experiment package root.
cd ".\Data Attribution Depletion Attack\code"

python Backdoor_Poisoning_Strategy_PPIS.py --ratio 0.01 --generation-only
python Backdoor_Poisoning_Strategy_PPIS_IRDS.py --poison-ratio 0.01 --epochs 100

python Attack_Success_Rate.py --clean-shapley <clean_csv> --poisoned-shapley <poison_csv> --mapping <mapping_csv>
python Gain_Stealing_Rate.py --clean-shapley <clean_csv> --poisoned-shapley <poison_csv> --mapping <mapping_csv>
python Gain_Volatility_Rate.py --clean-shapley <clean_csv> --poisoned-shapley <poison_csv> --mapping <mapping_csv>
python Loss_Amplifying_Rate.py --clean-shapley <clean_csv> --poisoned-shapley <poison_csv> --clean-loss-curve <clean_loss_csv> --poisoned-loss-curve <poison_loss_csv>
```

Command-line arguments differ across strategies. Run `python <script>.py --help` before starting an experiment. ImageNet-100 and ViT-B/16 experiments are computationally demanding; use `--max-train-samples`, a small number of `--epochs`, and fewer `--nc-steps` for an initial smoke test before launching a full run.

This code is intended for academic research and security evaluation. Its purpose is to expose potential risks in IRDS-based data attribution for reward allocation and data cleaning, and to support the development of more robust data attribution and defense mechanisms.
