# renlian

> 一个早期的 Python 人脸识别实验项目，基于 OpenCV、scikit-learn、TensorFlow / Keras。

[中文](README.md) | [English](README.en.md)

## 概述

本仓库用于演示从摄像头或图片采集人脸数据、训练模型、再进行简单识别的流程。项目偏学习和实验性质，不适合作为生产级身份认证系统。

## 功能

- 使用 OpenCV Haar Cascade 检测人脸。
- 支持采集和整理训练图片。
- 使用 TensorFlow / Keras 训练识别模型。
- 提供简单的人脸识别脚本。

## 依赖

- `opencv-python`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`

建议使用独立虚拟环境安装，避免和系统 Python 冲突。

## 基本流程

1. 准备 Python 环境并安装依赖。
2. 使用 `get_face2.py` 采集或整理人脸图片。
3. 使用 `face_train.py` 训练模型。
4. 使用 `Face_recognition.py` 运行识别。

## 局限性

- 这是学习项目，不提供生产级准确率、活体检测、权限隔离或审计能力。
- 依赖版本较旧，新的 Python / TensorFlow 环境可能需要手动调整。
- 人脸数据属于敏感个人信息，采集和使用前应获得明确授权。

## 隐私与安全

不要提交真实人脸数据、训练好的私人模型或包含个人身份信息的文件。公开使用人脸识别项目时，应遵守当地隐私、肖像权和数据保护要求。

## License

当前仓库尚未包含明确的 license 文件。正式分发或接受外部贡献前，建议补充。
