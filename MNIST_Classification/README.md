# MNIST Handwritten Digit Classification using Deep Learning

## Project Overview
This project implements a Deep Learning-based handwritten digit classification system using the MNIST dataset. The model is built using TensorFlow and Keras to classify grayscale images of handwritten digits (0–9).

## Objectives
- Learn image classification using neural networks
- Implement a Multilayer Perceptron (MLP)
- Perform data preprocessing and normalization
- Train, evaluate, and visualize model performance

## Dataset
- Name: MNIST
- Images: 28x28 grayscale
- Training samples: 60,000
- Testing samples: 10,000
- Classes: Digits 0–9

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure
```
mnist_classification.py
README.md
```

## Workflow
1. Import required libraries
2. Load MNIST dataset
3. Visualize sample images
4. Normalize pixel values
5. Build Sequential neural network
6. Compile the model
7. Train with validation split
8. Evaluate accuracy
9. Plot loss and accuracy graphs
10. Predict single image output

## Model Architecture
- Flatten Layer (28x28 → 784)
- Dense Layer (128 neurons, ReLU)
- Dense Layer (32 neurons, ReLU)
- Output Layer (10 neurons, Softmax)

## Model Compilation
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam
- Metric: Accuracy

## Results
The model achieves high accuracy on the test dataset and performs well for digit classification.

## How to Run
```bash
pip install tensorflow matplotlib scikit-learn
python mnist_classification.py
```

## Learning Outcomes
- Understanding neural networks for image data
- Hands-on experience with TensorFlow & Keras
- Importance of normalization and activation functions

## Future Improvements
- Use CNN instead of MLP
- Add Dropout layers
- Save and load trained models
- Deploy as a web application

## Author
Hardik Thapaliya

## License
This project is for educational purposes only.
