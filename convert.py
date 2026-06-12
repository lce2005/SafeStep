from pillow_heif import register_heif_opener
from PIL import Image
import os

register_heif_opener()

for root, dirs, files in os.walk('data'):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            img.save(filepath, 'JPEG')
            print(f"변환 완료: {filepath}")
        except Exception as e:
            print(f"스킵: {filepath} - {e}")
