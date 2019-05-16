import json
import os

import cv2
import numpy as np

IMAGE_SIZE = 64


# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape  # (237, 237, 3)

    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

        # RGB颜色
    BLACK = [0, 0, 0]

    # 边缘填充 0 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))


# 读取训练数据
images = []
labels = []


def read_path(path_name):
    for dir_item in os.listdir(path_name):

        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)

                # 放开这个代码，可以看到resize_image()函数的实际调用效果
                # cv2.imwrite('1.jpg', image)

                images.append(image)
                labels.append(path_name.split('\\')[-1])

    return images, labels


# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)
    print('labels:', labels)

    # 将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    # 我和室友两个人共600张图片，IMAGE_SIZE为64，故对我来说尺寸为1200 * 64 * 64 * 3
    # 图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)

    # 标注数据，'liuzhiyu'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹下是同学的，全部指定为1
    labels1 = list(set(labels))
    face_num = len(labels1)
    print('face_num:',face_num)
    num = [i for i in range(face_num)]
    contrast_table = dict(zip(num, labels1))
    with open('contrast_table', 'w') as f:
        f.write(json.dumps(contrast_table))
    # print('contrast_table:', contrast_table)
    for index, name in contrast_table.items():
        for i in range(len(labels)):
            if labels[i] == name:
                labels[i] = index
    # print(labels)
    labels = np.array(labels)

    return images, labels, face_num


# def load_dataset(path_name):
#     images, labels = read_path(path_name)
#     print('labels_name:', labels)
#
#     # 将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
#     # 我和室友两个人共600张图片，IMAGE_SIZE为64，故对我来说尺寸为1200 * 64 * 64 * 3
#     # 图片为64 * 64像素,一个像素3个颜色值(RGB)
#     images = np.array(images)
#
#     # 标注数据，'liuzhiyu'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹下是同学的，全部指定为1
#     contrast_table = {"0": "ljw", "1": "lzy", "2": "yzz", "3": "ldh", '4': 'wyz'}
#     for index, name in contrast_table.items():
#         for i in range(len(labels)):
#             if labels[i] == name:
#                 labels[i] = index
#     print('labels_number:', labels)
#     labels = np.array(labels)
#     return images, labels


if __name__ == '__main__':
    images, labels = load_dataset("C:\\Users\\84810\\Desktop\\csdn_renlian\\data")
