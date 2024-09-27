import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Define a function to create and compile a model with a given activation function
def create_model(activation_function):
    model = keras.Sequential([
        layers.Dense(128, activation=activation_function, input_shape=(28 * 28,)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# List of activation functions to test
activation_functions = ['sigmoid', 'tanh', 'relu']
results = {}

# Train and evaluate the model for each activation function
for activation in activation_functions:
    print(f'Training model with {activation} activation function...')
    model = create_model(activation)
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[activation] = test_acc
    print(f'Test accuracy with {activation}: {test_acc:.4f}')

# Plot the results
plt.bar(results.keys(), results.values())
plt.xlabel('Activation Functions')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Different Activation Functions')
plt.ylim([0.8, 1.0])
plt.show()
