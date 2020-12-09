'''Arquivo para extração da features'''

import tensorflow as tf
import numpy as np
import os

#Rodar a extração com um de cada vez
# directory = "../dataset_normalized/training_set" 
directory = "../dataset_normalized/validation_set"

res_model = tf.keras.applications.ResNet152(include_top=False, pooling='avg', weights='imagenet')

gen = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    batch_size=1,
    image_size=(355, 370),
    shuffle=False
)

X = []
y = []

for img_info in gen:
    label = img_info[1].numpy()[0]
    img = img_info[0].numpy()
    img = tf.keras.applications.resnet.preprocess_input(img)
    features = res_model.predict(img)
    features_reduce = features.squeeze()
    X.append(features_reduce)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Features de treino
# np.save("./dataset_featurized/resnet/features.npy", X)
# np.save("./dataset_featurized/resnet/labels.npy", y)

# Features de teste
np.save("./dataset_featurized/resnet/features_test.npy", X)
np.save("./dataset_featurized/resnet/labels_test.npy", y)
