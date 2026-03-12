"""
Reproduction of "Gatekeeper: Improving Model Cascades Through Confidence Tuning"
(Rabanser et al., NeurIPS 2025 / ICML Workshop 2025)

Paper: https://arxiv.org/abs/2502.19335

Key idea: Fine-tune M_S with a hybrid loss that:
  - Sharpens confidence on correct predictions (CE loss)
  - Flattens confidence on incorrect predictions (KL to uniform)
Then use max-softmax confidence as deferral signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import json
import os


# ============================================================
# Models
# ============================================================

class SmallCNN(nn.Module):
    """Custom small CNN (M_S) for CIFAR."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_large_model(num_classes=100):
    """ResNet-18 as M_L."""
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ============================================================
# Gatekeeper Loss (Equation 2 from the paper)
# ============================================================

class GatekeeperLoss(nn.Module):
    """
    L = α * L_corr + (1-α) * L_incorr
    
    L_corr = (1/N) * sum_{i: y_i == y_hat_i} CE(p_i, y_i)
    L_incorr = (1/N) * sum_{i: y_i != y_hat_i} KL(p_i || U)
    
    where U is the uniform distribution over C classes.
    """
    def __init__(self, alpha=0.5, num_classes=100):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        correct_mask = (preds == targets)
        incorrect_mask = ~correct_mask
        
        N = logits.size(0)
        
        # L_corr: CE on correct predictions
        if correct_mask.sum() > 0:
            l_corr = F.cross_entropy(logits[correct_mask], targets[correct_mask],
                                      reduction='sum') / N
        else:
            l_corr = torch.tensor(0.0, device=logits.device)
        
        # L_incorr: KL(p || U) on incorrect predictions
        if incorrect_mask.sum() > 0:
            log_probs = F.log_softmax(logits[incorrect_mask], dim=1)
            uniform = torch.full_like(log_probs, 1.0 / self.num_classes)
            # KL(p || U) = sum p * log(p/U) = sum p * (log_p - log(1/C))
            l_incorr = F.kl_div(uniform.log(), probs[incorrect_mask],
                                 reduction='sum') / N
            # Actually KL(p||q) = sum p * log(p/q), but F.kl_div expects log(q)
            # and computes sum(q * (log(q) - log(p))) by default...
            # Let's compute it manually for clarity:
            log_uniform = torch.full_like(log_probs, -np.log(self.num_classes))
            l_incorr = (probs[incorrect_mask] * (log_probs - log_uniform)).sum() / N
        else:
            l_incorr = torch.tensor(0.0, device=logits.device)
        
        loss = self.alpha * l_corr + (1 - self.alpha) * l_incorr
        return loss, l_corr.item(), l_incorr.item(), correct_mask.float().mean().item()


# ============================================================
# Deferral Evaluation Metrics
# ============================================================

def compute_deferral_metrics(small_probs, small_preds, large_preds, targets):
    """
    Compute deferral metrics from the paper:
    1. Distributional overlap s_o
    2. Deferral performance s_d
    3. AUROC for correct/incorrect separation
    
    Args:
        small_probs: (N, C) softmax outputs of M_S
        small_preds: (N,) predicted labels from M_S
        large_preds: (N,) predicted labels from M_L
        targets: (N,) true labels
    """
    # Max confidence as deferral signal (Eq. 5)
    confidences = small_probs.max(dim=1).values.numpy()
    correct = (small_preds == targets).numpy().astype(bool)
    
    # 1. AUROC: how well confidence separates correct/incorrect
    if correct.sum() > 0 and (~correct).sum() > 0:
        auroc = roc_auc_score(correct, confidences)
    else:
        auroc = float('nan')
    
    # 2. Deferral performance s_d (Eq. 8)
    # Sweep thresholds and compute joint accuracy at each deferral ratio
    acc_small = correct.mean()
    acc_large = (large_preds == targets).numpy().mean()
    
    thresholds = np.linspace(0, 1, 200)
    deferral_ratios = []
    joint_accs = []
    
    for tau in thresholds:
        defer_mask = confidences < tau  # defer if confidence below threshold
        deferral_ratio = defer_mask.mean()
        
        # Joint accuracy: use M_S where not deferred, M_L where deferred
        joint_correct = np.where(defer_mask,
                                  (large_preds == targets).numpy(),
                                  correct)
        joint_acc = joint_correct.mean()
        
        deferral_ratios.append(deferral_ratio)
        joint_accs.append(joint_acc)
    
    deferral_ratios = np.array(deferral_ratios)
    joint_accs = np.array(joint_accs)
    
    # Random deferral: linear interpolation from acc_small to acc_large
    acc_rand = acc_small + deferral_ratios * (acc_large - acc_small)
    
    # Ideal deferral: first defer the ones M_S gets wrong
    n_incorrect = (~correct).sum()
    n_total = len(correct)
    ideal_accs = []
    for r in deferral_ratios:
        n_defer = int(r * n_total)
        if n_defer <= n_incorrect:
            # Can fix some incorrect ones
            ideal_acc = acc_small + n_defer / n_total
        else:
            ideal_acc = acc_large  # All incorrect deferred + some correct
        ideal_accs.append(min(ideal_acc, acc_large))
    acc_ideal = np.array(ideal_accs)
    
    # s_d = A_perf / A_useful
    a_perf = np.trapz(joint_accs - acc_rand, deferral_ratios)
    a_useful = np.trapz(acc_ideal - acc_rand, deferral_ratios)
    s_d = a_perf / a_useful if a_useful > 0 else 0.0
    
    # 3. Distributional overlap (simplified using histogram overlap)
    conf_correct = confidences[correct]
    conf_incorrect = confidences[~correct]
    bins = np.linspace(0, 1, 100)
    if len(conf_correct) > 0 and len(conf_incorrect) > 0:
        hist_c, _ = np.histogram(conf_correct, bins=bins, density=True)
        hist_i, _ = np.histogram(conf_incorrect, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        s_o = np.sum(np.minimum(hist_c, hist_i)) * bin_width
    else:
        s_o = float('nan')
    
    return {
        'auroc': auroc,
        's_d': s_d,
        's_o': s_o,
        'acc_small': acc_small,
        'acc_large': acc_large,
        'deferral_ratios': deferral_ratios.tolist(),
        'joint_accs': joint_accs.tolist(),
    }


# ============================================================
# Data
# ============================================================

def get_data(dataset='cifar100', batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                   download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                  download=True, transform=transform_test)
        num_classes = 100
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform_test)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


# ============================================================
# Training Functions
# ============================================================

def pretrain(model, trainloader, testloader, device, epochs=50, lr=0.1, name="Model"):
    """Standard training with CE loss."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for images, labels in tqdm(trainloader, desc=f"{name} Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        
        scheduler.step()
        
        # Test
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                test_correct += (model(images).argmax(1) == labels).sum().item()
                test_total += labels.size(0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {name} Epoch {epoch+1}: Train {100*correct/total:.1f}% "
                  f"| Test {100*test_correct/test_total:.1f}% | Loss {running_loss/len(trainloader):.3f}")
    
    return model


def gatekeeper_finetune(model, trainloader, testloader, device, alpha=0.5,
                         num_classes=100, epochs=20, lr=0.001):
    """Fine-tune M_S with Gatekeeper loss."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    gk_loss = GatekeeperLoss(alpha=alpha, num_classes=num_classes)
    
    print(f"\n=== Gatekeeper Fine-tuning (α={alpha}) ===")
    for epoch in range(epochs):
        model.train()
        total_loss, total_corr, total_incorr, total_acc = 0, 0, 0, 0
        n_batches = 0
        for images, labels in tqdm(trainloader, desc=f"GK Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss, l_c, l_i, batch_acc = gk_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_corr += l_c
            total_incorr += l_i
            total_acc += batch_acc
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f} "
                  f"L_corr={total_corr/n_batches:.4f} L_incorr={total_incorr/n_batches:.4f} "
                  f"batch_acc={100*total_acc/n_batches:.1f}%")
    
    return model


def evaluate_cascade(small_model, large_model, testloader, device):
    """Evaluate the cascade deferral performance."""
    small_model.eval()
    large_model.eval()
    
    all_small_probs = []
    all_small_preds = []
    all_large_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            small_out = F.softmax(small_model(images), dim=1)
            large_out = large_model(images).argmax(dim=1)
            
            all_small_probs.append(small_out.cpu())
            all_small_preds.append(small_out.argmax(dim=1).cpu())
            all_large_preds.append(large_out.cpu())
            all_targets.append(labels.cpu())
    
    small_probs = torch.cat(all_small_probs)
    small_preds = torch.cat(all_small_preds)
    large_preds = torch.cat(all_large_preds)
    targets = torch.cat(all_targets)
    
    return compute_deferral_metrics(small_probs, small_preds, large_preds, targets)


# ============================================================
# Plotting
# ============================================================

def plot_results(all_results, dataset):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    alphas = sorted([k for k in all_results.keys() if k != 'baseline'], key=lambda x: float(x))
    
    # 1. Distributional Overlap s_o
    s_o_vals = [all_results[a]['s_o'] for a in alphas]
    axes[0].bar([str(a) for a in alphas], s_o_vals, color='steelblue')
    axes[0].set_xlabel('α')
    axes[0].set_ylabel('Distributional Overlap (s_o) ↓')
    axes[0].set_title('Correct/Incorrect Confidence Overlap')
    
    # 2. Deferral Performance s_d
    s_d_vals = [all_results[a]['s_d'] for a in alphas]
    axes[1].bar([str(a) for a in alphas], s_d_vals, color='seagreen')
    axes[1].set_xlabel('α')
    axes[1].set_ylabel('Deferral Performance (s_d) ↑')
    axes[1].set_title('Deferral Performance')
    
    # 3. Small Model Accuracy
    acc_vals = [all_results[a]['acc_small'] * 100 for a in alphas]
    axes[2].bar([str(a) for a in alphas], acc_vals, color='coral')
    axes[2].set_xlabel('α')
    axes[2].set_ylabel('M_S Accuracy (%)')
    axes[2].set_title('Small Model Accuracy')
    
    plt.suptitle(f'Gatekeeper Results on {dataset.upper()}')
    plt.tight_layout()
    plt.savefig(f'results_{dataset}.png', dpi=150)
    print(f"Plot saved to results_{dataset}.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Gatekeeper Reproduction')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--phase', default='all', choices=['pretrain', 'gatekeeper', 'evaluate', 'all'])
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='Alpha values for Gatekeeper')
    parser.add_argument('--pretrain-epochs', type=int, default=50)
    parser.add_argument('--gk-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainloader, testloader, num_classes = get_data(args.dataset, args.batch_size)
    os.makedirs('checkpoints', exist_ok=True)
    
    if args.phase in ['pretrain', 'all']:
        # Pre-train M_L (ResNet-18)
        print("\n=== Pre-training M_L (ResNet-18) ===")
        large_model = get_large_model(num_classes).to(device)
        pretrain(large_model, trainloader, testloader, device,
                 epochs=args.pretrain_epochs, lr=0.1, name="M_L")
        torch.save(large_model.state_dict(), f'checkpoints/large_{args.dataset}.pt')
        
        # Pre-train M_S (Small CNN)
        print("\n=== Pre-training M_S (SmallCNN) ===")
        small_model = SmallCNN(num_classes).to(device)
        pretrain(small_model, trainloader, testloader, device,
                 epochs=args.pretrain_epochs, lr=0.1, name="M_S")
        torch.save(small_model.state_dict(), f'checkpoints/small_{args.dataset}.pt')
    
    if args.phase in ['gatekeeper', 'evaluate', 'all']:
        # Load pre-trained models
        large_model = get_large_model(num_classes).to(device)
        large_model.load_state_dict(torch.load(f'checkpoints/large_{args.dataset}.pt',
                                                weights_only=True))
        large_model.eval()
    
    if args.phase in ['gatekeeper', 'all']:
        # Evaluate baseline (no Gatekeeper)
        print("\n=== Baseline Evaluation (no Gatekeeper) ===")
        small_baseline = SmallCNN(num_classes).to(device)
        small_baseline.load_state_dict(torch.load(f'checkpoints/small_{args.dataset}.pt',
                                                    weights_only=True))
        baseline_metrics = evaluate_cascade(small_baseline, large_model, testloader, device)
        print(f"  Baseline: acc_S={baseline_metrics['acc_small']:.3f} "
              f"acc_L={baseline_metrics['acc_large']:.3f} "
              f"s_d={baseline_metrics['s_d']:.3f} s_o={baseline_metrics['s_o']:.3f} "
              f"AUROC={baseline_metrics['auroc']:.3f}")
        
        all_results = {'baseline': baseline_metrics}
        
        # Fine-tune with Gatekeeper at different alphas
        for alpha in args.alpha:
            print(f"\n{'='*50}")
            print(f"  Gatekeeper with α={alpha}")
            print(f"{'='*50}")
            
            # Load fresh copy of pre-trained M_S
            small_model = SmallCNN(num_classes).to(device)
            small_model.load_state_dict(torch.load(f'checkpoints/small_{args.dataset}.pt',
                                                     weights_only=True))
            
            # Fine-tune with Gatekeeper
            small_model = gatekeeper_finetune(small_model, trainloader, testloader, device,
                                               alpha=alpha, num_classes=num_classes,
                                               epochs=args.gk_epochs, lr=0.001)
            
            # Save fine-tuned model
            torch.save(small_model.state_dict(),
                       f'checkpoints/small_gk_{args.dataset}_a{alpha}.pt')
            
            # Evaluate cascade
            metrics = evaluate_cascade(small_model, large_model, testloader, device)
            all_results[alpha] = metrics
            
            print(f"  α={alpha}: acc_S={metrics['acc_small']:.3f} "
                  f"s_d={metrics['s_d']:.3f} s_o={metrics['s_o']:.3f} "
                  f"AUROC={metrics['auroc']:.3f}")
        
        # Print summary
        print("\n" + "="*70)
        print(f"{'Method':<20} {'acc(M_S)':<12} {'s_d ↑':<10} {'s_o ↓':<10} {'AUROC ↑':<10}")
        print("="*70)
        for key in all_results:
            m = all_results[key]
            label = f"Baseline" if key == 'baseline' else f"GK α={key}"
            print(f"{label:<20} {m['acc_small']:.3f}        {m['s_d']:.3f}     "
                  f"{m['s_o']:.3f}     {m['auroc']:.3f}")
        
        # Save results
        # Remove non-serializable items
        save_results = {}
        for k, v in all_results.items():
            save_results[str(k)] = {kk: vv for kk, vv in v.items()
                                     if kk not in ['deferral_ratios', 'joint_accs']}
        with open(f'results_{args.dataset}.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Plot
        plot_alphas = {k: v for k, v in all_results.items() if k != 'baseline'}
        plot_alphas['baseline'] = all_results['baseline']
        plot_results(plot_alphas, args.dataset)


if __name__ == '__main__':
    main()
