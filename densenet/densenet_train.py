'''Arquivo para testes de abordagens'''

import tensorflow as tf
from dense_block import dense_block, transition_block

train_directory = "../dataset_normalized/training_set"
test_directory = "../dataset_normalized/validation_set"

dense_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((355, 370, 3)))

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

# tunned_model = dense_model.output

dense_model.layers.pop()
dense_model.layers.pop()

bn_axis = 3
tunned_model = transition_block(dense_model.layers[-3].output, 0.5, name='pool5')
tunned_model = dense_block(tunned_model, 32, name='conv6')
tunned_model = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(tunned_model)
tunned_model = tf.keras.layers.Activation('relu', name='relu')(tunned_model)

tunned_model = tf.keras.layers.GlobalAveragePooling2D() (tunned_model)
tunned_model = tf.keras.layers.Flatten() (tunned_model)
tunned_model = tf.keras.layers.Dense(256, activation="relu") (tunned_model)
tunned_model = tf.keras.layers.Dropout(0.5) (tunned_model)
tunned_model = tf.keras.layers.Dense(5, activation="softmax") (tunned_model)

model = tf.keras.Model(inputs=dense_model.input, outputs=tunned_model)

for layer in dense_model.layers:
	layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_steps = 6177 / 3
valid_steps = 1544 / 8
test_steps = 856 / 8

model.fit(train_gen, epochs=20, steps_per_epoch=train_steps, validation_data=valid_gen, validation_steps=valid_steps)

# model.save("./dense_tuned.h5")

labels = model.predict(test_gen)

results = model.evaluate(test_gen)
print("test loss, test acc:", results)
