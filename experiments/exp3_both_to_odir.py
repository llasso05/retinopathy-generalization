import sys
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim

# Adjust import paths depending on how the script is run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import get_experiment_config
from datasets.loaders import APTOSDataset, MessidorDataset, ODIRDataset
from preprocessing.transforms import get_transforms
from models import get_model
from training.engine import train_model, validate
from evaluation.metrics import calculate_metrics, plot_confusion_matrix, save_metrics

def main():
    config = get_experiment_config('exp3_both_to_odir')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================
    # 1. Load Datasets
    # ==========================
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # Train on APTOS + Messidor
    ds_aptos_train = APTOSDataset(config['dataset_paths']['aptos'], transform=train_transform, split='train')
    ds_messidor_train = MessidorDataset(config['dataset_paths']['messidor'], transform=train_transform, split='train')
    train_dataset = ConcatDataset([ds_aptos_train, ds_messidor_train])
    
    # Evaluate on ODIR (Generalization)
    val_dataset = ODIRDataset(config['dataset_paths']['odir'], transform=val_transform, split='train') # Note: ODIR test split is currently empty/unmapped in loaders
    
    # Self-Test on APTOS + Messidor
    st_aptos = APTOSDataset(config['dataset_paths']['aptos'], transform=val_transform, split='test')
    st_messidor = MessidorDataset(config['dataset_paths']['messidor'], transform=val_transform, split='test')
    self_test_dataset = ConcatDataset([st_aptos, st_messidor])

    if len(train_dataset) == 0:
        print("Error: Training dataset is empty.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
                            shuffle=False, num_workers=config['training']['num_workers'])
    self_test_loader = DataLoader(self_test_dataset, batch_size=config['training']['batch_size'], 
                                  shuffle=False, num_workers=config['training']['num_workers'])

    # ==========================
    # 2. Setup Model & Training
    # ==========================
    model = get_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    exp_dir = os.path.join(config['output_dir'], 'exp3_both_odir')
    checkpoint_path = os.path.join(exp_dir, 'best_model.pt')
    os.makedirs(exp_dir, exist_ok=True)

    # ==========================
    # 3. Train Model
    # ==========================
    print("Starting Training on APTOS + Messidor...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config['training']['num_epochs'], config['training']['patience'],
        checkpoint_path, device
    )

    # ==========================
    # 4. Final Evaluation
    # ==========================
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    # Self-Test
    print("Evaluating on APTOS + Messidor Test Set (Self-Test)...")
    self_loss, y_pred_self, y_true_self, y_prob_self = validate(model, self_test_loader, criterion, device)
    self_metrics = calculate_metrics(y_true_self, y_pred_self, y_prob_self)
    self_metrics['val_loss'] = self_loss
    save_metrics(self_metrics, os.path.join(exp_dir, 'self_test_metrics.json'))
    plot_confusion_matrix(y_true_self, y_pred_self, class_names, 
                          save_path=os.path.join(exp_dir, 'self_test_confusion_matrix.png'))

    # Generalization on ODIR
    print("Evaluating on ODIR (Generalization)...")
    val_loss, y_pred, y_true, y_prob = validate(model, val_loader, criterion, device)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics['val_loss'] = val_loss
    
    # Save results
    save_metrics(metrics, os.path.join(exp_dir, 'metrics.json'))
    plot_confusion_matrix(y_true, y_pred, class_names, 
                          save_path=os.path.join(exp_dir, 'confusion_matrix.png'))
                          
    print(f"Experiment completed. Results saved to {exp_dir}")

if __name__ == '__main__':
    main()
