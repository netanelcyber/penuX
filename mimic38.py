import torch
import torch.nn.functional as F

def compute_multiclass_pr_auc(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 12):
    """
    Computes Macro-Averaged Precision-Recall AUC purely in PyTorch.
    Evaluates One-vs-Rest for each class to accurately reflect minority class performance.
    """
    # 1. Convert raw logits to probabilities
    probs = F.softmax(logits, dim=1)
    
    pr_aucs = []
    class_metrics = {}
    
    for c in range(num_classes):
        # 2. Isolate the current class (One-vs-Rest)
        true_binary = (labels == c).float()
        total_positives = true_binary.sum()
        
        # Skip if the class is completely absent in the current evaluation batch
        if total_positives == 0:
            continue
            
        # Extract the model's predicted probabilities for this specific class
        class_probs = probs[:, c]
        
        # 3. Sort probabilities in descending order
        sorted_probs, indices = torch.sort(class_probs, descending=True)
        sorted_true = true_binary[indices]
        
        # 4. Cumulative sums to calculate True Positives (TP) and False Positives (FP) dynamically
        tps = torch.cumsum(sorted_true, dim=0)
        fps = torch.cumsum(1.0 - sorted_true, dim=0)
        
        # 5. Calculate Precision and Recall at every threshold
        # Add 1e-8 to the denominator to prevent division by zero
        precision = tps / (tps + fps + 1e-8)
        recall = tps / total_positives
        
        # 6. Anchor the curves at Recall=0
        # Precision at Recall=0 is conventionally set to 1.0
        precision = torch.cat([torch.tensor([1.0], device=logits.device), precision])
        recall = torch.cat([torch.tensor([0.0], device=logits.device), recall])
        
        # 7. Calculate Area Under the Curve (AUC) using the Trapezoidal Rule
        # AUC = sum( ΔRecall * Average_Precision )
        delta_recall = recall[1:] - recall[:-1]
        mean_precision = (precision[1:] + precision[:-1]) / 2.0
        
        auc = torch.sum(delta_recall * mean_precision).item()
        
        pr_aucs.append(auc)
        class_metrics[c] = auc
        
    # 8. Calculate Macro-Average across all present classes
    macro_pr_auc = sum(pr_aucs) / len(pr_aucs) if pr_aucs else 0.0
    
    return macro_pr_auc, class_metrics

# ==========================================
# Integration into the Evaluation Loop
# ==========================================
# Example of how you call this inside evaluate_model():
#
# macro_pr_auc, per_class_auc = compute_multiclass_pr_auc(all_logits, all_labels, num_classes=12)
# print(f"[*] Macro PR-AUC: {macro_pr_auc:.4f}")
# 
# # Specifically highlighting a minority class (e.g., Class 2 = MRSA)
# if 2 in per_class_auc:
#     print(f"[*] MRSA (Class 2) PR-AUC: {per_class_auc[2]:.4f}")
