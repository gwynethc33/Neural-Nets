# Sign Language Digit Recognition Using MLP Neural Network
This project uses a Multi-Layer Perceptron (MLP) neural network to classify hand sign digits from image data. The images are represented as pixel values in CSV files, and the goal is to train a model that accurately predicts the corresponding sign class.

## Features

- Normalizes pixel values (scales 0-255 to 0-1).
- Splits the training data into training and validation sets (80%/20%).
- Builds an MLP classifier with two hidden layers (100 and 50 neurons).
- Trains the model on the training set.
- Evaluates model accuracy on training, validation, and test sets.
- Computes per-class accuracy on the validation set.
- Generates and prints a confusion matrix for the test set.
- Prints detailed information about model architecture and data sizes.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
