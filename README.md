# renlian

> 一个早期的 Python 人脸识别实验项目，基于 OpenCV、scikit-learn、TensorFlow / Keras。

## Overview

这个仓库用于演示从摄像头或图片采集人脸数据、训练模型、再进行简单识别的流程。项目偏学习和实验性质，不适合作为生产级身份认证系统。

## Features

- 使用 OpenCV Haar Cascade 检测人脸。
- 支持采集和整理训练图片。
- 使用 TensorFlow / Keras 训练识别模型。
- 提供简单的人脸识别脚本。

## Requirements

核心依赖：

- `opencv-python`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`

建议使用独立虚拟环境安装依赖，避免和系统 Python 冲突。

## Project Structure

```text
renlian/
├── data/                               # 训练或测试数据
├── model/                              # 训练输出模型
├── haarcascade_frontalface_default.xml # OpenCV 人脸检测模型
├── get_face2.py                        # 人脸采集
├── face_train.py                       # 模型训练
├── Face_recognition.py                 # 人脸识别
├── load_data.py                        # 数据加载
└── rename.py                           # 数据文件整理
```

## Basic Workflow

1. 准备 Python 环境并安装依赖。
2. 使用 `get_face2.py` 采集或整理人脸图片。
3. 使用 `face_train.py` 训练模型。
4. 使用 `Face_recognition.py` 运行识别。

## Limitations

- 这是学习项目，不提供生产级准确率、活体检测、权限隔离或审计能力。
- 依赖版本较旧，新的 Python / TensorFlow 环境可能需要手动调整。
- 人脸数据属于敏感个人信息，采集和使用前应获得明确授权。

## Security and Privacy

不要提交真实人脸数据、训练好的私人模型或包含个人身份信息的文件。公开使用人脸识别项目时，应遵守当地隐私、肖像权和数据保护要求。

## License

No license file is currently included. Add an explicit license before redistributing or accepting contributions.
