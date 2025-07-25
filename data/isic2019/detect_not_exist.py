import os
import pandas as pd

def detect_missing_images(
    csv_path: str,
    image_dir: str,
    image_col: str = "image",
    image_ext: str = ".jpg",
    save_path: str = None,
):
    """
    Detect missing images listed in a CSV file and optionally save a filtered CSV.
    """
    df = pd.read_csv(csv_path)
    
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in CSV file.")

    image_names = df[image_col].values
    missing_files = []
    exist_mask = []

    for name in image_names:
        image_path = os.path.join(image_dir, name + image_ext)
        if not os.path.exists(image_path):
            missing_files.append(name)
            exist_mask.append(False)
        else:
            exist_mask.append(True)

    print(f"Total images listed: {len(image_names)}")
    print(f"Missing images     : {len(missing_files)}")

    if save_path:
        df_cleaned = df[exist_mask]
        df_cleaned.to_csv(save_path, index=False)
        print(f"Saved cleaned CSV to: {save_path}")

    if "UNK" in df.columns:
        print("⚠️  Detected 'UNK' column. Ensure it is handled properly in your training script.")

    return missing_files

# Example usage
if __name__ == "__main__":
    root = "/home/work/DATA2/isic2019"
    mode = "test"  # or "test"
    csv_path = os.path.join(root, f"{mode}.csv")
    image_dir = os.path.join(root, "ISIC_2019_Test_Input")
    save_path = os.path.join(root, f"{mode}_filtered.csv")

    detect_missing_images(csv_path, image_dir, save_path=save_path)
