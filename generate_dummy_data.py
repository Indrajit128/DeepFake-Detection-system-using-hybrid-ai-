import os
import numpy as np
from PIL import Image
import config

def generate_dataset(base_dir=config.DATA_DIR, total_images=2041):
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # 50/50 split roughly
    num_real = total_images // 2
    num_fake = total_images - num_real

    print(f"Generating {num_real} 'real' images and {num_fake} 'fake' images...")

    for i in range(num_real):
        # Generate random noise image for "real"
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(real_dir, f"real_{i:04d}.jpg"))

    for i in range(num_fake):
        # Generate random noise image for "fake" but slightly different distribution
        img = np.random.randint(50, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(fake_dir, f"fake_{i:04d}.jpg"))

    print("Dummy dataset generated successfully.")

if __name__ == "__main__":
    generate_dataset()

