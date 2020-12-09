'''Arquivo para testes de abordagens'''

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from residual_block import stack_block
import tensorflow as tf
import numpy as np
import os

train_directory = "../dataset_normalized/training_set"
test_directory = "../dataset_normalized/validation_set"

res_model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((355, 370, 3)))

train_gen = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=3,
    image_size=(355, 370),
    validation_split=0.2,
    seed=2222,
    subset="training",
)

valid_gen = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=8,
    image_size=(355, 370),
    validation_split=0.2,
    seed=2222,
    subset="validation",
)

test_gen = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=1,
    image_size=(355, 370)
)

# tunned_model = stack_block(res_model.output, 1024, 3, name="conv6") 
tunned_model = res_model.output
tunned_model = tf.keras.layers.GlobalAveragePooling2D() (tunned_model)
tunned_model = tf.keras.layers.Flatten() (tunned_model)
tunned_model = tf.keras.layers.Dense(256, activation="relu") (tunned_model)
tunned_model = tf.keras.layers.Dropout(0.5) (tunned_model)
tunned_model = tf.keras.layers.Dense(5, activation="softmax") (tunned_model)

model = tf.keras.Model(inputs=res_model.input, outputs=tunned_model)

for layer in res_model.layers[30:]:
	layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_steps = 6177 / 3
valid_steps = 1544 / 8
test_steps = 856 / 8

model.fit(train_gen, epochs=10, steps_per_epoch=train_steps, validation_data=valid_gen, validation_steps=valid_steps)

labels = model.predict(test_gen)

results = model.evaluate(test_gen)
print("test loss, test acc:", results)

