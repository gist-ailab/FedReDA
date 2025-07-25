import os
import numpy as np
from PIL import Image
import glob

dataset = 'ham10000'
mode = 'train'
dir_path = f"/home/work/Workspaces/yunjae_heo/FedLNL/data/{dataset}/{mode}"

label_dict = {'AKIEC':0, 'BCC':1, 'BKL':2, 'DF':3, 'MEL':4, 'NV':5, 'VASC':6}

img_list = glob.glob(os.path.join(dir_path, '*', '*'))
img_list = [img for img in img_list if img.endswith(".png") or img.endswith(".jpg")]
print(len(img_list))

img_list_np = []
label_list_np = []

for i in img_list:
    img = Image.open(i).convert("RGB")
    img = img.resize((224, 224))  # 💥 꼭 필요!
    img_array = np.array(img)
    img_list_np.append(img_array)

    key = i.split('/')[-2]
    if key not in label_dict:
        print(f"[❗] Unknown label found: '{key}' in path: {i}")
    else:
        label_list_np.append(label_dict[key])

# ✅ 정규 numpy 배열로 저장 (shape: [N, H, W, 3])
img_np = np.stack(img_list_np).astype(np.uint8)  # 또는 float32
label_np = np.array(label_list_np, dtype=np.int64)

# ✅ 저장
np.save(f'/home/work/Workspaces/yunjae_heo/FedLNL/data/{dataset}/{mode}_images.npy', img_np)
np.save(f'/home/work/Workspaces/yunjae_heo/FedLNL/data/{dataset}/{mode}_labels.npy', label_np)

print(len(label_list_np))
print(label_list_np[-3:])
print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")

if __name__ == '__main__':
    dataset = 'ham10000'
    mode = 'train'
    npy_img_path = f'/home/work/Workspaces/yunjae_heo/FedLNL/data/{dataset}/{mode}_images.npy'
    npy_label_path = f'/home/work/Workspaces/yunjae_heo/FedLNL/data/{dataset}/{mode}_labels.npy'

    # ✅ Numpy 파일 불러오기
    img_np = np.load(npy_img_path)  # shape: (N, 224, 224, 3)
    label_np = np.load(npy_label_path)  # shape: (N,)

    print(f"Loaded image shape: {img_np.shape}")
    print(f"Loaded label shape: {label_np.shape}")
    print(f"First label: {label_np[0]}")

    # ✅ 첫 번째 이미지 확인
    img = img_np[0]  # shape: (224, 224, 3)
    print(f"First image min/max: {img.min()}/{img.max()}, dtype: {img.dtype}")

    # ✅ 이미지 저장
    img_pil = Image.fromarray(img.astype(np.uint8))  # 혹시 float로 저장되었을 경우 대비
    img_pil.save("input_tensor.png")
    print("Saved first image to input_tensor.png ✅")