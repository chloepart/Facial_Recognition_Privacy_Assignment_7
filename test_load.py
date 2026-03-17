import os
from PIL import Image

path = os.path.expanduser('~/Downloads/att_faces')
print(f'Dataset path: {path}')
print(f'Path exists: {os.path.exists(path)}')

test_img = os.path.join(path, 's1', '1.pgm')
print(f'Test image: {test_img}')
print(f'Test exists: {os.path.exists(test_img)}')

try:
    img = Image.open(test_img)
    print(f'Success! Size: {img.size}, Mode: {img.mode}')
except Exception as e:
    print(f'Error: {e}')
