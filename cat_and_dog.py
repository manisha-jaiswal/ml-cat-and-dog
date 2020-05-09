import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Male ' , 'Female' ]


_URL = 'https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification.zip'

path_to_zip = tf.keras.utils.get_file('male_and_female.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'male_and_female_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_male_dir = os.path.join(train_dir, 'male')  # directory with our training cat pictures
train_female_dir = os.path.join(train_dir, 'female')  # directory with our training dog pictures
validation_male_dir = os.path.join(validation_dir, 'male')  # directory with our validation cat pictures
validation_female_dir = os.path.join(validation_dir, 'female')  # directory with our validation dog pictures
num_male_tr = len(os.listdir(train_male_dir))
num_female_tr = len(os.listdir(train_female_dir))

num_male_val = len(os.listdir(validation_male_dir))
num_female_val = len(os.listdir(validation_female_dir))

total_train = num_male_tr + num_female_tr
total_val = num_male_val + num_female_val

print('total training male images:', num_male_tr)
print('total training female images:', num_female_tr)

print('total validation male images:', num_male_val)
print('total validation female images:', num_female_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict(sample_training_images)
print(predictions)

for i in range(10):
	plt.grid(False)
	plt.imshow(sample_training_images[i], cmap=plt.cm.binary)
	plt.show()
