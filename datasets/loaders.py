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
        except Exception:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.images))
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

class APTOSDataset(BaseDRDataset):
    """
    Loader for the APTOS 2019 Blindness Detection dataset.
    """
    def __init__(self, data_dir, transform=None, split='train'):
        self.split = split
        self.csv_name = f"{split}.csv"
        self.img_folder_name = f"{split}_images"
        super().__init__(data_dir, transform)

    def _load_data(self):
        csv_path = os.path.join(self.data_dir, self.csv_name)
        img_folder = os.path.join(self.data_dir, self.img_folder_name)
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}. Using empty dataset.")
            return
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(img_folder, f"{row['id_code']}.png")
            
            if os.path.exists(img_path):
                self.images.append(img_path)
                label = row.get('diagnosis', -1) 
                self.labels.append(int(label))
            else:
                # print(f"Warning: image not found {img_path}")
                pass

class MessidorDataset(BaseDRDataset):
    """
    Loader for Messidor dataset.
    """
    def __init__(self, data_dir, transform=None, split='train'):
        self.split = split
        self.csv_name = f"{split}.csv"
        self.img_folder_name = split
        super().__init__(data_dir, transform)

    def _load_data(self):
        csv_path = os.path.join(self.data_dir, self.csv_name)
        img_folder = os.path.join(self.data_dir, self.img_folder_name)
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}. Using empty dataset.")
            return
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # In the new structure, Image column has the .tif extension
            img_path = os.path.join(img_folder, str(row['Image']))
                
            # Map 'Id' column to labels (0-4)
            label = int(row.get('Id', -1))
                
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(int(label))
            else:
                # print(f"Warning: image not found {img_path}")
                pass

class ODIRDataset(BaseDRDataset):
    """
    Loader for ODIR dataset.
    Maps keywords to 0-4 DR severity levels.
    """
    def __init__(self, data_dir, transform=None, split='train'):
        self.split = split
        # ODIR seems to have full_df.csv as the main meta file
        self.csv_name = 'full_df.csv'
        self.img_folder_name = 'Training Images' if split == 'train' else 'Testing Images'
        super().__init__(data_dir, transform)

    def _extract_label(self, diagnosis):
        diagnosis = str(diagnosis).lower()
        if 'proliferative diabetic retinopathy' in diagnosis:
            return 4
        elif 'severe nonproliferative diabetic retinopathy' in diagnosis:
            return 3
        elif 'moderate nonproliferative diabetic retinopathy' in diagnosis:
            return 2
        elif 'mild nonproliferative diabetic retinopathy' in diagnosis:
            return 1
        return 0 # Normal or other disease

    def _load_data(self):
        csv_path = os.path.join(self.data_dir, self.csv_name)
        img_folder = os.path.join(self.data_dir, self.img_folder_name)
        
        if not os.path.exists(csv_path):
            print(f"Warning: Annotation file not found at {csv_path}. Using empty dataset.")
            return
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Check Left Eye
            left_img = os.path.join(img_folder, str(row['Left-Fundus']))
            if os.path.exists(left_img):
                label = self._extract_label(row['Left-Diagnostic Keywords'])
                # Only add if it's DR or explicitly Normal fundus
                if label > 0 or 'normal fundus' in str(row['Left-Diagnostic Keywords']).lower():
                    self.images.append(left_img)
                    self.labels.append(label)

            # Check Right Eye
            right_img = os.path.join(img_folder, str(row['Right-Fundus']))
            if os.path.exists(right_img):
                label = self._extract_label(row['Right-Diagnostic Keywords'])
                if label > 0 or 'normal fundus' in str(row['Right-Diagnostic Keywords']).lower():
                    self.images.append(right_img)
                    self.labels.append(label)
