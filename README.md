# 🐍 Venomous vs Non-Venomous Snake Identifier

A deep learning image classification system that identifies whether a snake is venomous or non-venomous from a photograph. Built as a portfolio project for COM 763 - Advanced Machine Learning.

## 🔗 Live Demo

**Try it here:** https://snake-venom-classifier-efhjv9hjemlntf3eyrn2gj.streamlit.app

⚠️ **Disclaimer:** This is an academic project for demonstration purposes only. Do not use this tool to make real-world safety decisions around snakes. Always treat unidentified snakes as potentially dangerous and consult local wildlife experts.

## 📌 Project Overview

This project compares three approaches to binary image classification (Venomous / Non-Venomous) using a dataset of Indian snake species:

1. **Baseline CNN** — a convolutional neural network trained from scratch
2. **MobileNetV2** — transfer learning using ImageNet pretrained weights
3. **EfficientNetB0** — transfer learning using ImageNet pretrained weights

## 📊 Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN (scratch) | 55% | 0.54 |
| MobileNetV2 | 83% | 0.82 |
| **EfficientNetB0 (final model)** | **87%** | **0.87** |

EfficientNetB0 was selected as the final deployed model based on the strongest and most balanced performance across both classes.

## 🗂️ Dataset

[Snake Dataset - India](https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india) — images of Indian snake species, pre-labeled and split into train/test sets across two classes: Venomous and Non-Venomous.

## 🛠️ Tech Stack

- **Python 3.10**
- **TensorFlow / Keras 2.15** — model development and training
- **Streamlit** — web app interface and deployment
- **scikit-learn** — evaluation metrics
- **Matplotlib / Seaborn** — visualizations

## 📁 Repository Structure
