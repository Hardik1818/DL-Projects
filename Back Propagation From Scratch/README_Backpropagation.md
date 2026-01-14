# Backpropagation from Scratch – Classification & Regression

## Project Overview
This project demonstrates **manual implementation of backpropagation** for both **classification** and **regression** problems using **NumPy**, without relying on high-level deep learning frameworks.

Two Jupyter notebooks are included:
- `backpropagation_classification.ipynb`
- `backpropagation_regression.ipynb`

The goal is to deeply understand how **forward propagation, loss computation, and backpropagation** work mathematically and programmatically.

---

## Objectives
- Implement neural networks **from scratch**
- Understand forward propagation step-by-step
- Derive and implement backpropagation equations
- Compare backpropagation behavior in classification vs regression
- Strengthen exam and viva understanding of neural networks

---

## Technologies Used
- Python
- NumPy
- Pandas
- Jupyter Notebook

---

## File Description

### 1️⃣ backpropagation_classification.ipynb
This notebook implements backpropagation for a **classification problem**.

#### Key Components
- Data loading and preprocessing
- Parameter initialization (weights & biases)
- Forward propagation using activation functions
- Loss computation (classification loss)
- Backpropagation using chain rule
- Weight and bias updates using gradient descent

#### Concepts Covered
- Sigmoid / Softmax activation
- Classification loss
- Gradient computation
- Parameter updates
- Iterative training loop

---

### 2️⃣ backpropagation_regression.ipynb
This notebook implements backpropagation for a **regression problem**.

#### Key Components
- Dataset preparation
- Linear activation in output layer
- Mean Squared Error (MSE) loss
- Backpropagation for regression
- Gradient descent optimization

#### Concepts Covered
- Linear output neurons
- Regression loss (MSE)
- Error propagation
- Weight updates

---

## Backpropagation Workflow (Common to Both)

1. Initialize weights and biases
2. Perform forward propagation
3. Compute loss
4. Compute gradients using backpropagation
5. Update parameters
6. Repeat for multiple epochs

---

## Key Differences: Classification vs Regression

| Aspect | Classification | Regression |
|------|---------------|------------|
| Output Activation | Sigmoid / Softmax | Linear |
| Loss Function | Cross-Entropy | Mean Squared Error |
| Output | Class labels | Continuous values |

---

## Learning Outcomes
- Clear understanding of how backpropagation works internally
- Ability to derive gradients manually
- Strong conceptual foundation for deep learning frameworks
- Improved confidence for exams and viva

---

## Academic Use
This project is **highly suitable for**:
- Deep Learning coursework
- Viva examinations
- Conceptual demonstrations
- Interview preparation (DL fundamentals)

---

## Author
Hardik Thapaliya

---

## License
This project is for **educational and academic purposes only**.
