# Gatekeeper Reproduction

Reproduction of "Gatekeeper: Improving Model Cascades Through Confidence Tuning" (Rabanser et al., 2025)

Paper: https://arxiv.org/abs/2502.19335

## Method

Gatekeeper is a hybrid loss for fine-tuning a small model M_S in a cascade setup:

```
L = α * L_corr + (1-α) * L_incorr
```

- `L_corr`: CE loss on correctly predicted samples (sharpen confidence)
- `L_incorr`: KL(p || Uniform) on incorrectly predicted samples (flatten confidence)
- `α`: trade-off parameter (lower = more aggressive deferral calibration)

## Experiments

### Image Classification (Encoder-only)
- **Datasets**: CIFAR-10, CIFAR-100, TinyImageNet200
- **M_S**: Custom CNN / MobileNetV3-Small
- **M_L**: ResNet-18 / ResNet-50

### Usage

```bash
# Train baseline models
python main.py --phase pretrain --dataset cifar100

# Fine-tune with Gatekeeper
python main.py --phase gatekeeper --dataset cifar100 --alpha 0.3

# Evaluate cascade deferral
python main.py --phase evaluate --dataset cifar100
```
