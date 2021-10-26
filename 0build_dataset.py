import os
import glob
import random
import errno
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
import cv2
import vgg

eps = 1e-8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

feature_extractor = vgg.VGG19()

input_root = "F:/ISTD_Dataset/train/train_A/"
mask_root = "F:/ISTD_Dataset/train/train_shadow/"
gt_root = "F:/ISTD_Dataset/train/train_C/"

input_dataset_root = []
mask_dataset_root = []
gt_dataset_root = []
for img_path in glob.glob(os.path.join(input_root, '*.png')):
    basename = os.path.basename(img_path)
    original_img_path = os.path.join(input_root, basename)
    mask_img_path = os.path.join(mask_root, basename)
    gt_img_path = os.path.join(gt_root, basename)

    input_dataset_root.append(original_img_path)
    mask_dataset_root.append(mask_img_path)
    gt_dataset_root.append(gt_img_path)

generated_dataset_shadow2non_match = []
generated_dataset_shadow2non_non_match = []
generated_dataset_non2shaodw_match = []
generated_dataset_non2shaodw_non_match = []
generated_dataset_shadow2shadow_match = []
generated_dataset_shadow2shadow_non_match = []

value1 = 0
value2 = 0
value3 = 0
value4 = 0

for i in range(len(input_dataset_root)):
    image = Image.open(input_dataset_root[i]).convert('RGB').resize((416, 416))
    mask = Image.open(mask_dataset_root[i]).convert('L').resize((416, 416))
    gt = Image.open(gt_dataset_root[i]).convert('RGB').resize((416, 416))

    non_shadow_feature = []
    shadow_feature = []
    features = []
    is_shadow = []
    for y in range(0, 416, 32):
        for x in range(0, 416, 32):
            gt_patch = gt.crop((x, y, x+32, y+32))
            gt_patch = F.to_tensor(gt_patch)
            gt_patch = F.normalize(gt_patch, mean, std).unsqueeze(0)
            feature_temp = feature_extractor(gt_patch)["relu5_4"].view(-1).numpy()
            features.append(feature_temp)
            mask_patch = mask.crop((x, y, x+32, y+32))
            mask_patch = np.array(mask_patch).astype(np.float32) / 255.0
            sum_mask = mask_patch.sum()
            if sum_mask > 512:
                is_shadow.append(1)
                non_shadow_feature.append(np.zeros(2048).astype(np.float32))
                shadow_feature.append(feature_temp.astype(np.float32))
            else:
                is_shadow.append(0)
                non_shadow_feature.append(feature_temp.astype(np.float32))
                shadow_feature.append(np.zeros(2048).astype(np.float32))

    features = torch.from_numpy(np.array(features))                                       # 169 2048
    non_shadow_feature = torch.from_numpy(np.array(non_shadow_feature).transpose(1, 0))   # 2048 169
    shadow_feature = torch.from_numpy(np.array(shadow_feature).transpose(1, 0))   # 2048 144

    features = (features / (features.norm(dim=1, keepdim=True) + eps)).unsqueeze(dim=0)
    non_shadow_feature = (non_shadow_feature / (non_shadow_feature.norm(dim=0, keepdim=True) + eps)).unsqueeze(dim=0)
    shadow_feature = (shadow_feature / (shadow_feature.norm(dim=0, keepdim=True) + eps)).unsqueeze(dim=0)

    correlation = torch.bmm(features, non_shadow_feature)                                 # 169 169
    correlation, indexes = torch.sort(correlation, dim=2, descending=True)
    correlation2 = torch.bmm(features, shadow_feature)                                    # 169 169
    correlation2, indexes2 = torch.sort(correlation2, dim=2, descending=True)
    # for j in range(144):
    #     if is_shadow[j] == 1:
    #         print(correlation[0, j])
    for j in range(169):
        if is_shadow[j] == 1:
            index = 0
            while (index < 5) and (correlation[0, j, index] > 0.8):     # shadow 2 non-shadow  match
                instance = (
                    input_dataset_root[i],                    # image root
                    int(j),                                        # patch a
                    int(indexes[0, j, index]),                     # patch b
                    2,                                        # pair type
                    1                                         # correlation degree
                )
                generated_dataset_shadow2non_match.append(instance)
                instance = (
                    input_dataset_root[i],  # image root
                    int(indexes[0, j, index]),  # patch a
                    int(j),  # patch b
                    0,  # pair type
                    1  # correlation degree
                )
                generated_dataset_non2shaodw_match.append(instance)
                index += 1
            index = 143
            flag = 0
            while (correlation[0, j, index] < 0.45) and (flag < 5):                   # shadow 2 non-shadow  non-match
                if is_shadow[index] == 0:
                    instance = (
                        input_dataset_root[i],  # image root
                        int(j),  # patch a
                        int(indexes[0, j, index]),  # patch b
                        2,  # pair type
                        0  # correlation degree
                    )
                    generated_dataset_shadow2non_non_match.append(instance)
                    instance = (
                        input_dataset_root[i],  # image root
                        int(indexes[0, j, index]),  # patch a
                        int(j),  # patch b
                        0,  # pair type
                        0  # correlation degree
                    )
                    generated_dataset_non2shaodw_non_match.append(instance)
                    flag += 1
                index -= 1
            index = 0
            flag = 0
            while (flag < 5) and (correlation2[0, j, index] > 0.8):
                if is_shadow[index] == 1:
                    instance = (
                        input_dataset_root[i],  # image root
                        int(j),  # patch a
                        int(indexes2[0, j, index]),  # patch b
                        1,  # pair type
                        1  # correlation degree
                    )
                    generated_dataset_shadow2shadow_match.append(instance)
                    flag += 1
                index += 1
            index = 143
            flag = 0
            while (flag < 5) and (correlation2[0, j, index] < 0.45):
                if is_shadow[index] == 1:
                    instance = (
                        input_dataset_root[i],  # image root
                        int(j),  # patch a
                        int(indexes2[0, j, index]),  # patch b
                        1,  # pair type
                        0  # correlation degree
                    )
                    generated_dataset_shadow2shadow_non_match.append(instance)
                    flag += 1
                index -= 1

    print(i, ":", input_dataset_root[i])
    print("shadow2non     match      :", len(generated_dataset_shadow2non_match) - value1)
    print("shadow2non     non-match  :", len(generated_dataset_shadow2non_non_match) - value2)
    print("shadow2shadow  match  :", len(generated_dataset_shadow2shadow_match) - value3)
    print("shadow2shadow  match  :", len(generated_dataset_shadow2shadow_non_match) - value4)
    value1 = len(generated_dataset_shadow2non_match)
    value2 = len(generated_dataset_shadow2non_non_match)
    value3 = len(generated_dataset_shadow2shadow_match)
    value4 = len(generated_dataset_shadow2shadow_non_match)

# dataset_match = generated_dataset_shadow2non_match + \
#           generated_dataset_non2shaodw_match + generated_dataset_shadow2shadow_match
#
# dataset_non_match = generated_dataset_shadow2non_non_match + generated_dataset_non2shaodw_non_match  + generated_dataset_shadow2shadow_non_match

generated_dataset_shadow2non_match = np.array(generated_dataset_shadow2non_match)
generated_dataset_non2shaodw_match = np.array(generated_dataset_non2shaodw_match)
generated_dataset_shadow2shadow_match = np.array(generated_dataset_shadow2shadow_match)
generated_dataset_shadow2non_non_match = np.array(generated_dataset_shadow2non_non_match)
generated_dataset_non2shaodw_non_match = np.array(generated_dataset_non2shaodw_non_match)
generated_dataset_shadow2shadow_non_match = np.array(generated_dataset_shadow2shadow_non_match)

with open("E:/generated_dataset_shadow2non_match.pt", 'wb') as f:
    torch.save(generated_dataset_shadow2non_match, f)

with open("E:/generated_dataset_non2shaodw_match.pt", 'wb') as f:
    torch.save(generated_dataset_non2shaodw_match, f)

with open("E:/generated_dataset_shadow2shadow_match.pt", 'wb') as f:
    torch.save(generated_dataset_shadow2shadow_match, f)

with open("E:/generated_dataset_shadow2non_non_match.pt", 'wb') as f:
    torch.save(generated_dataset_shadow2non_non_match, f)

with open("E:/generated_dataset_non2shaodw_non_match.pt", 'wb') as f:
    torch.save(generated_dataset_non2shaodw_non_match, f)

with open("E:/generated_dataset_shadow2shadow_non_match.pt", 'wb') as f:
    torch.save(generated_dataset_shadow2shadow_non_match, f)




    # image = np.array(image, dtype='float32') / 255.0
