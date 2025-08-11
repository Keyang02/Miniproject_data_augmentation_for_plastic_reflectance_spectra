import os

root_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(root_dir, 'gif'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'log'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'loss'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'nets'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'synthetic_data'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'vs_img'), exist_ok=True)