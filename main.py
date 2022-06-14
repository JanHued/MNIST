from tensorflow import keras

data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

number_of_categories = 10
train_labels_vectorized = keras.utils.to_categorical(train_labels, number_of_categories)
test_labels_vectorized = keras.utils.to_categorical(test_labels, number_of_categories)

pixel_max_value = 255
train_images_normalized = train_images / pixel_max_value
test_images_normalized = test_images / pixel_max_value

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(number_of_categories, activation='sigmoid')
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(train_images_normalized, train_labels_vectorized, epochs=20)

eval_loss, eval_accuracy = model.evaluate(test_images_normalized, test_labels_vectorized)
