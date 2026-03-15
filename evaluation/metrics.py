import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics.
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted classes.
        y_prob (np.array): Predicted probabilities for ROC AUC.
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and roc_auc.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Use macro average for multi-class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # ROC AUC requires probabilities
    # Use 'ovo' or 'ovr' for multi-class
    try:
        # One-vs-Rest for multi-class
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except ValueError:
        # Fails if not all classes are present in y_true, fallback to NaN
        roc_auc = float('nan')
        
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plots and optionally saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def save_metrics(metrics, save_path):
    """
    Saves metrics dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
