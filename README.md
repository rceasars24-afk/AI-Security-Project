# AI Security Capstone Project

## Overview
This project demonstrates how machine learning systems can be vulnerable to adversarial attacks and how those attacks can be defended against.

A Convolutional Neural Network (CNN) was trained using the MNIST handwritten digit dataset. The model was then attacked using the Fast Gradient Sign Method (FGSM), causing a major drop in accuracy. Finally, adversarial training was used to improve the model's robustness.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib

---

## Dataset

MNIST handwritten digit dataset

- 60,000 training images
- 10,000 testing images
- 10 classes (digits 0–9)

---

## Project Stages

### 1. Baseline Model
A CNN model was trained on clean MNIST data.

**Accuracy:** ~ 96-98%

---

### 2. Adversarial Attack (FGSM)

FGSM was used to generate adversarial examples that slightly modified the input images.

**Accuracy Under Attack:** ~ 4-7%

---

### 3. Defense (Adversarial Training)

The model was retrained using adversarial examples.

**Defended Accuracy:** ~ 95-99%

---

## Results

| Stage | Accuracy |
|-------|----------|
| Baseline | 98.22% |
| FGSM Attack | 5.66% |
| Defense | 98.83% |

---

## Visualization

Project output graph:

results/model_results.png

---

## How to Run

### Clone Repository

git clone (https://github.com/yourusername/AI-Security-Project.git)

### Install Dependencies

pip install -r requirements.txt

### Run Project

python main.py

---

## Author

Rommell St. Preux