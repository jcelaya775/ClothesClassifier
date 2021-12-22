import tensorflow as ts
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# shrink down data to between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)), # input layer (flattened to an array)
	keras.layers.Dense(128, activation="relu"), # fully connected hidden layer w/ 128 neurons & ReLU activation function
	keras.layers.Dense(10, activation="softmax") # fully connected output layer (activation function compresses input to be between 0 and 1) and gives probablity for each class
	])

# set up parameters for model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epochs are how many times the model trains on the same set of images (but in different orders) 
model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)
# prediction = model.predict([test_images[3]]) -> to predict one image 

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual Item: " + class_names[test_labels[i]])
	plt.title("Predicted Item: " + class_names[np.argmax(prediction[i])])
	plt.show()