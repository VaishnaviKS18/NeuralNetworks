import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the data
file_path = "spamm.txt"  # Make sure the path to your file is correct
data = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])

# Check for any missing values and drop them
data.dropna(subset=["text"], inplace=True)

# Encode labels: spam = 1, ham = 0
data['label'] = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data into features and target
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the vectorizer and fit_transform the training data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Convert sparse matrix to dense matrix
X_train_dense = X_train_vectorized.toarray()

# Vectorize the test data and convert to dense
X_test_vectorized = vectorizer.transform(X_test)
X_test_dense = X_test_vectorized.toarray()

# Initialize the DNN model
model = Sequential()

# Add layers to the model
model.add(Dense(64, input_dim=X_train_dense.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to avoid overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_dense, y_train, epochs=5, batch_size=32, validation_data=(X_test_dense, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_dense, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save the model architecture, weights, and vectorizer in .pkl format
model_config = model.get_config()  # Model architecture
model_weights = model.get_weights()  # Model weights

# Save model architecture and weights as a dictionary in .pkl format
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump({'model_config': model_config, 'model_weights': model_weights}, f)

# Save the vectorizer in .pkl format
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model architecture, weights, and vectorizer have been saved successfully in .pkl format.")
