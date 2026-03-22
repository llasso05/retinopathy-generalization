import sys
import os
import torch
from torch.utils.data import DataLoader
from torch import nn

# Adjust import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import get_experiment_config
from datasets.loaders import APTOSDataset, MessidorDataset
from preprocessing.transforms import get_transforms
from models import get_model
from training.engine import validate
from evaluation.metrics import calculate_metrics, plot_confusion_matrix, save_metrics

def main():
    config = get_experiment_config('exp2_messidor_to_aptos')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_transform = get_transforms(config, is_training=False)
    
    # Load model
    model = get_model(config).to(device)
    exp_dir = os.path.join(config['output_dir'], 'exp2_messidor_aptos')
    checkpoint_path = os.path.join(exp_dir, 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Datasets
    val_dataset = APTOSDataset(config['dataset_paths']['aptos'], transform=val_transform, split='test')
    self_test_dataset = MessidorDataset(config['dataset_paths']['messidor'], transform=val_transform, split='test')
    
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    self_test_loader = DataLoader(self_test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    # 1. Self-Test
    print("Evaluating on Messidor Test Set (Self-Test)...")
    self_loss, y_pred_self, y_true_self, y_prob_self = validate(model, self_test_loader, criterion, device)
    self_metrics = calculate_metrics(y_true_self, y_pred_self, y_prob_self)
    save_metrics(self_metrics, os.path.join(exp_dir, 'self_test_metrics.json'))
    plot_confusion_matrix(y_true_self, y_pred_self, class_names, save_path=os.path.join(exp_dir, 'self_test_confusion_matrix.png'))

    # 2. Generalization
    print("Evaluating on APTOS (Generalization)...")
    val_loss, y_pred, y_true, y_prob = validate(model, val_loader, criterion, device)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    save_metrics(metrics, os.path.join(exp_dir, 'metrics.json'))
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(exp_dir, 'confusion_matrix.png'))
    
    print(f"Evaluation completed for Exp 2. Results saved to {exp_dir}")

if __name__ == '__main__':
    main()
