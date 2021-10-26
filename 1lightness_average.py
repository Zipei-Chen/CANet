import PIL.Image as Image
import numpy as np
import cv2
import glob
import os


def make_dataset():
    dataset = []
    original_img_rpath = 'F:/ISTD_Dataset/train/train_A'
    for img_path in glob.glob(os.path.join(original_img_rpath, '*.png')):
        basename = os.path.basename(img_path)
        original_img_path = os.path.join(original_img_rpath, basename)
        dataset.append(original_img_path)
    return dataset


def average(image):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            sum1 += image[0][i][j]
            sum2 += image[1][i][j]
            sum3 += image[2][i][j]
    return sum1 / ((i+1)*(j+1)), sum2 / ((i+1)*(j+1)), sum3 / ((i+1)*(j+1))


def average_image(image, x, y, r):
    sum1 = 0
    sum2 = 0
    sum3 = 0

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            sum1 += image[0][x + i][y + j]
            sum2 += image[1][x + i][y + j]
            sum3 += image[2][x + i][y + j]

    sum1 = sum1 / ((2 * r + 1) * (2 * r + 1))
    sum2 = sum2 / ((2 * r + 1) * (2 * r + 1))
    sum3 = sum3 / ((2 * r + 1) * (2 * r + 1))
    return sum1, sum2, sum3


def reduce_avg_channel(image, x, y, r, avg, c):
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            image[x+i][y+j] = image[x+i][y+j] - avg + c

    return image


Input = make_dataset()
for k in range(len(Input)):
    name = Input[k]
    print(k)
    print(name)

    image = Image.open(name).convert('RGB')
    image = np.array(image, dtype='float32') / 255.0

    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  #rgb to lab

    image = image.transpose(2, 0, 1)    # h w c to c h w

    average1, average2, average3 = average(image)

    L = image[0, :, :]
    A = image[1, :, :]
    B = image[2, :, :]

    r = 1   # radius of block

    for i in range(1, L.shape[0]-1, 3):
        for j in range(1, L.shape[1]-1, 3):
            avg1, avg2, avg3 = average_image(image, i, j, r)
            L = reduce_avg_channel(L, i, j, r, avg1, average1)
            B = reduce_avg_channel(B, i, j, r, avg3, average3)

    image[0, :, :] = L
    image[2, :, :] = B

    image = image.transpose(1, 2, 0)

    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB) * 255.0

    image = Image.fromarray(np.uint8(image))
    image.save(os.path.join('F:/ISTD_Dataset/new_test2/', os.path.basename(name)))