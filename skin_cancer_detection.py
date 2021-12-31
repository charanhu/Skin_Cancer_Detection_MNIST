# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

# reference: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/discussion/183083
# Data: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/layers/merging_layers/add/
# https://keras.io/api/layers/convolution_layers/convolution2d
# https://keras.io/api/layers/convolution_layers/convolution2d
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import Sequential

classes = {
    0: ("actinic keratoses and intraepithelial carcinomae(Cancer)"),
    1: ("basal cell carcinoma(Cancer)"),
    2: ("benign keratosis-like lesions(Non-Cancerous)"),
    3: ("dermatofibroma(Non-Cancerous)"),
    4: ("melanocytic nevi(Non-Cancerous)"),
    5: ("pyogenic granulomas and hemorrhage(Can lead to cancer)"),
    6: ("melanoma(Cancer)"),
}


model = Sequential()
model.add(
    Conv2D(
        16,
        kernel_size=(3, 3),
        input_shape=(28, 28, 3),
        activation="relu",
        padding="same",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(7, activation="softmax"))
model.summary()
model.load_weights("best_model.h5")
