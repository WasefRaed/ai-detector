# src/evaluation/evaluate.py
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def full_evaluation(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs.cpu().numpy())

    # Classification report
    print(classification_report(all_labels, all_preds,
          target_names=["Human", "Hybrid", "AI"]))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human", "Hybrid", "AI"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")

    # ROC-AUC (multiclass)
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    print(f"ROC-AUC (OvR): {auc:.4f}")