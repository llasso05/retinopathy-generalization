import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.loaders import APTOSDataset
from models import get_model
from training.engine import validate
from evaluation.metrics import calculate_metrics, plot_confusion_matrix, save_metrics
from utils.config import get_experiment_config

def evaluate_aptos_test():
    # 1. Load configuration
    config = get_experiment_config('aptos_test_eval')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Define transforms (must match training)
    val_transforms = transforms.Compose([
        transforms.Resize((config['preprocessing']['image_size'], config['preprocessing']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['preprocessing']['normalize_mean'],
                             std=config['preprocessing']['normalize_std'])
    ])

    # 3. Load APTOS Test Dataset
    aptos_path = config['dataset_paths']['aptos']
    test_dataset = APTOSDataset(
        aptos_path, 
        transform=val_transforms, 
        csv_name='test.csv.xls', 
        img_folder='test_images'
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training']['num_workers']
    )
    print(f"Loaded {len(test_dataset)} test images from APTOS.")

    # 4. Load Model
    model = get_model(config).to(device)
    model_path = os.path.join(config['output_dir'], 'exp1_aptos_messidor', 'best_model.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # 5. Evaluate
    criterion = nn.CrossEntropyLoss()
    print("Evaluating on APTOS test set...")
    _, y_pred, y_true, y_prob = validate(model, test_loader, criterion, device)

    # 6. Calculate and Save Metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print("\nAPTOS Test Set Metrics:")
    print(metrics)

    results_dir = os.path.join(config['output_dir'], 'aptos_test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_path = os.path.join(results_dir, 'metrics_aptos_test.json')
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    # 7. Generate Confusion Matrix
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    cm_path = os.path.join(results_dir, 'confusion_matrix_aptos_test.png')
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    evaluate_aptos_test()
