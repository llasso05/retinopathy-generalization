import os
import csv
from collections import Counter

def check_aptos(data_dir, split='train'):
    print(f"\n--- Checking APTOS {split} ---")
    csv_path = os.path.join(data_dir, f"{split}.csv")
    img_folder = os.path.join(data_dir, f"{split}_images")
    
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    
    found = 0
    missing = 0
    labels = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = f"{row['id_code']}.png"
            img_path = os.path.join(img_folder, img_name)
            if os.path.exists(img_path):
                found += 1
                if 'diagnosis' in row:
                    labels.append(row['diagnosis'])
            else:
                missing += 1
    
    print(f"Found: {found}")
    print(f"Missing: {missing}")
    if labels:
        print(f"Label distribution: {dict(sorted(Counter(labels).items()))}")

def check_messidor(data_dir, split='train'):
    print(f"\n--- Checking Messidor {split} ---")
    csv_path = os.path.join(data_dir, f"{split}.csv")
    img_folder = os.path.join(data_dir, split)
    
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    
    found = 0
    missing = 0
    labels = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean possible trailing spaces in keys
            row = {k.strip(): v for k, v in row.items()}
            img_name = row.get('Image')
            if not img_name:
                continue
            img_path = os.path.join(img_folder, img_name)
            if os.path.exists(img_path):
                found += 1
                if 'Id' in row:
                    labels.append(row['Id'])
            else:
                missing += 1
    
    print(f"Found: {found}")
    print(f"Missing: {missing}")
    if labels:
        print(f"Label distribution: {dict(sorted(Counter(labels).items()))}")

def extract_odir_label(diagnosis):
    diagnosis = str(diagnosis).lower()
    if 'proliferative diabetic retinopathy' in diagnosis: return '4'
    if 'severe nonproliferative diabetic retinopathy' in diagnosis: return '3'
    if 'moderate nonproliferative diabetic retinopathy' in diagnosis: return '2'
    if 'mild nonproliferative diabetic retinopathy' in diagnosis: return '1'
    if 'normal fundus' in diagnosis: return '0'
    return None

def check_odir(data_dir, split='train'):
    print(f"\n--- Checking ODIR {split} ---")
    csv_path = os.path.join(data_dir, "full_df.csv")
    img_folder = os.path.join(data_dir, 'Training Images' if split == 'train' else 'Testing Images')
    
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    
    found = 0
    missing = 0
    labels = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check Left Eye
            l_img_name = row.get('Left-Fundus')
            if l_img_name:
                l_img = os.path.join(img_folder, l_img_name)
                if os.path.exists(l_img):
                    found += 1
                    lbl = extract_odir_label(row['Left-Diagnostic Keywords'])
                    if lbl: labels.append(lbl)
                else:
                    missing += 1
            
            # Check Right Eye
            r_img_name = row.get('Right-Fundus')
            if r_img_name:
                r_img = os.path.join(img_folder, r_img_name)
                if os.path.exists(r_img):
                    found += 1
                    lbl = extract_odir_label(row['Right-Diagnostic Keywords'])
                    if lbl: labels.append(lbl)
                else:
                    missing += 1
    
    print(f"Found: {found}")
    print(f"Missing: {missing}")
    if labels:
        print(f"Label distribution (DR+Normal): {dict(sorted(Counter(labels).items()))}")

def main():
    base_dir = "datasets"
    check_aptos(os.path.join(base_dir, "aptos"), "train")
    check_aptos(os.path.join(base_dir, "aptos"), "test")
    
    check_messidor(os.path.join(base_dir, "messidor"), "train")
    check_messidor(os.path.join(base_dir, "messidor"), "test")
    
    check_odir(os.path.join(base_dir, "odir"), "train")
    check_odir(os.path.join(base_dir, "odir"), "test")

if __name__ == "__main__":
    main()
