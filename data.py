import kagglehub
import os
os.makedirs("data/raw/", exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("kartik2112/fraud-detection", output_dir="data/raw/")

print("Path to dataset files:", path)