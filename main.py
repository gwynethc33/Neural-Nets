import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0  # Normalize the data
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
y_test = test_data['class']   # Target (first column)
X_test = X_test / 255.0  # Normalize the data
# Manually select the first 130 samples for the training set

# Step 1: Split the data into training and validation sets
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=.2 , random_state=42, stratify=y_train)

# Create the MLPClassifier model
neural_net_model = MLPClassifier(hidden_layer_sizes=(100,50), random_state=42, tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Training set size: {len(y_train)}")
print(f"Layer sizes: {layer_size_str}")

# Predict the labels for training, validation, and test datasets
y_pred_train = neural_net_model.predict(X_train)
y_pred_validate = neural_net_model.predict(X_validate)
y_pred_test = neural_net_model.predict(X_test)

# Calculate and print accuracy for training, validation, and test sets
train_accuracy = neural_net_model.score(X_train, y_train)
val_accuracy = neural_net_model.score(X_validate, y_validate)
test_accuracy = neural_net_model.score(X_test, y_test)


# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct predictions for each class on the validation set
for true, pred in zip(y_validate, y_pred_validate):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# Count correct predictions for each class on the training set
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# Calculate and print accuracy for each class on the validation set
print("\nAccuracy per Class (Validation Set):")
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")

# Overall validation accuracy
overall_validation_accuracy = overall_correct / len(y_validate) * 100
print(f"Overall Validation Accuracy: {overall_validation_accuracy:3.1f}%") 

# Overall training accuracy
overall_training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")

# Step 6: Print confusion matrix for the test set
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix for Test Set:")
class_ids = sorted(set(y_test))

for i, class_id in enumerate(class_ids):
    print(f"Class {class_id}: ", end="")
    print("\t".join(str(conf_matrix[i, j]) for j in range(len(class_ids))))

print("\n The model had the most difficulty identifying Class 3 and Class 7.")

