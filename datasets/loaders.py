import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BaseDRDataset(Dataset):
    """
    Base class for Diabetic Retinopathy datasets.
    Standardizes label mapping (0-4).
    0: No DR
    1: Mild
    2: Moderate
    3: Severe
    4: Proliferative DR
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        raise NotImplementedError("Subclasses must implement _load_data")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image via PIL
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle the error appropriately in custom cases
            raise e
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

class APTOSDataset(BaseDRDataset):
    """
    Loader for the APTOS 2019 Blindness Detection dataset.
    Expecting a 'train.csv' file with 'id_code' and 'diagnosis' columns.
    Images should be in an 'images' subfolder or similar.
    """
    def _load_data(self):
        csv_path = os.path.join(self.data_dir, 'train.csv')
        img_folder = os.path.join(self.data_dir, 'train_images')
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}. Using empty dataset.")
            return
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(img_folder, f"{row['id_code']}.png")
            self.images.append(img_path)
            # APTOS labels are already 0-4
            self.labels.append(int(row['diagnosis']))


class MessidorDataset(BaseDRDataset):
    """
    Loader for Messidor dataset.
    Assuming a generalized CSV format with 'image_id' and 'retinopathy_grade'.
    """
    def _load_data(self):
        csv_path = os.path.join(self.data_dir, 'messidor_data.csv')
        img_folder = os.path.join(self.data_dir, 'images')
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}. Using empty dataset.")
            return
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(img_folder, str(row['image_id']))
            if not img_path.endswith('.jpg') and not img_path.endswith('.png'):
                img_path += '.jpg' # Assume jpg if extension missing
                
            self.images.append(img_path)
            # Standardize Messidor (0-3 scale typically, map to 0-4 if needed or keep as is, assuming 0-3 maps suitably or is provided standard)
            label = int(row['retinopathy_grade'])
            # Example mapping if Messidor is 0-3:
            # 0: No DR, 1: Mild, 2: Moderate/Severe, 3: Proliferative.
            # You might need to adjust mapping based on exact data format used.
            if label == 3: 
                label = 4 # Map to Proliferative DR
            elif label == 2:
                label = 2 # Moderate
                
            self.labels.append(label)

class ODIRDataset(BaseDRDataset):
    """
    Loader for the Ocular Disease Intelligent Recognition (ODIR) dataset.
    Focuses only on the DR labels for this context.
    """
    def _load_data(self):
        csv_path = os.path.join(self.data_dir, 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
        img_folder = os.path.join(self.data_dir, 'ODIR-5K_Training_Dataset')
        
        if not os.path.exists(csv_path):
            print(f"Warning: Annotation file not found at {csv_path}. Using empty dataset.")
            return
            
        # ODIR is typically an Excel sheet
        df = pd.read_excel(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(img_folder, str(row['Left-Fundus']))
            
            # ODIR has complex labels (N, D, G, C, A, H, M, O). 'D' is Diabetic Retinopathy.
            # ODIR doesn't strictly have 0-4 severity naturally in the core label, 
            # it only gives presence/absence of DR ('D' keyword) usually, BUT there are diagnostic keywords.
            # For this simplified loader, we'll dummy map it.
            # If cross-dataset evaluation needs 5 classes, you might need severity extraction from keywords.
            
            # simplified dummy logic:
            diagnosis = str(row['Left-Diagnostic Keywords']).lower()
            label = 0
            if 'proliferative diabetic retinopathy' in diagnosis:
                label = 4
            elif 'severe nonproliferative diabetic retinopathy' in diagnosis:
                label = 3
            elif 'moderate nonproliferative diabetic retinopathy' in diagnosis:
                label = 2
            elif 'mild nonproliferative diabetic retinopathy' in diagnosis:
                label = 1
            elif 'diabetic retinopathy' in diagnosis:
                label = 2 # Default fallback
            
            # We add left eye if it's DR labeled or Normal
            if label > 0 or 'normal fundus' in diagnosis:
                self.images.append(img_path)
                self.labels.append(label)
                
            # Could repeat for Right Eye (row['Right-Fundus']) appropriately.
