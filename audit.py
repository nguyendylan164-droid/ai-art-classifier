# Data analysis

import os
import hashlib
from collections import Counter
from PIL import Image

def audit_folder(folder_name):
    folder_path = os.path.join(".", "Art", folder_name)
    sizes = Counter()
    formats = Counter()
    hashes = set()
    duplicates = 0

    print(f"\nAuditing: {folder_name}")
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Skip hidden Mac files like .DS_Store
        if not os.path.isfile(filepath) or filename.startswith('.'):
            continue

        try:
            # 1. Check sizes and formats
            with Image.open(filepath) as img:
                sizes[img.size] += 1
                formats[img.format] += 1
            
            # 2. Check for exact duplicate files using a hash
            with open(filepath, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hashes:
                    duplicates += 1
                else:
                    hashes.add(file_hash)
        except Exception:
            print(f"Could not read {filename}. Is it a valid image?")

    print(f"Total Unique Images: {len(hashes)}")
    print(f"Exact Duplicates Found: {duplicates}")
    print(f"File Formats: {dict(formats)}")
    print(f"Top 3 Image Sizes: {sizes.most_common(3)}")

# Run the audit on both folders
audit_folder("RealArt")
audit_folder("AIArtData")