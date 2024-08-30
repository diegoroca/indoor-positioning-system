import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot_utils import save_and_open_plot
import seaborn as sns

# PREPROCESSING DATA ###############################################################################

# Directory containing the CSV files
data_dir = "Training Data"

# Initialize lists to hold the features and labels
features = []
labels = []

# Iterate over each file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        # Extract the label from the filename (e.g., 'A1' from 'A1.csv')
        label = filename.split('.')[0]
        
        # Load the CSV file into a DataFrame
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath) #asumes that the firs row is the header by default
        
        # Opt-in to the future behavior
        pd.set_option('future.no_silent_downcasting', True)
        # Replace 'x' values with -100
        df.replace('x', -100, inplace=True)
        
        # Convert the DataFrame to a NumPy array and remove the first two columns (SSID, MAC Address)
        feature_array = df.iloc[:, 2:].to_numpy()
        
        # Append each row to the features list and the corresponding label to the labels list
        for row in feature_array:
            features.append(row)
            labels.append(label)

# Convert features and labels lists to NumPy arrays
X = np.array(features, dtype=float)
Y = np.array(labels)

# Print the shape of the features and labels to verify
print(f"Features shape: {X.shape}")
print(f"Labels shape: {Y.shape}")

# Splitting into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.24, random_state=46)

# TRAINING KNN #####################################################################################

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)

# PERFORMING KNN FOR ONE SAMPLE ####################################################################

# Select a sample from the test set
sample_index = 50  # Index of the sample you want to predict
sample_to_predict = X_test[sample_index].reshape(1, -1)  # Reshape to 2D array

# Predict for one sample using the trained model
prediction = knn.predict(sample_to_predict)
true_label = Y_test[sample_index]
print(f'The predicted label for the test sample is: {prediction[0]}')
print(f'The true label for the test sample is: {true_label}')

# Choose indices of the 3 features to plot
feature_indices = [2, 3, 4]  # Replace with the indices of your features

# Extract the 3 chosen features from training data
X_train_3d = X_train[:, feature_indices]
X_test_3d = X_test[:, feature_indices]

# Extract the specific sample from the test set
sample_to_predict_3d = sample_to_predict[:, feature_indices].flatten()

# Encode labels as numeric values
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot training data
scatter = ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], c=Y_train_encoded, cmap='viridis', label='Training Data')

# Plot the sample to predict
ax.scatter(sample_to_predict_3d[0], sample_to_predict_3d[1], sample_to_predict_3d[2], c='red', s=50, label='Sample to Predict')

# Add labels and legend
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Create a color bar to show the mapping from numeric values to labels
cbar = plt.colorbar(scatter)
cbar.set_label('Labels')
cbar.set_ticks(np.arange(len(label_encoder.classes_)))
cbar.set_ticklabels(label_encoder.classes_)

ax.legend()

save_and_open_plot("prediction.png");

# TESTING ACCURACY #################################################################################

# Predict for all testing samples
Y_pred = knn.predict(X_test)

# Print accuracy score 0 to 1 scale
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Compute the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)

# Encode labels as numeric values for visualization
label_encoder = LabelEncoder()
Y_test_encoded = label_encoder.fit_transform(Y_test)
Y_pred_encoded = label_encoder.transform(Y_pred)

# Compute confusion matrix with encoded labels
cm_encoded = confusion_matrix(Y_test_encoded, Y_pred_encoded)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_encoded, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

save_and_open_plot("confution_matrix.png");

