import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds

mnist = tf.keras.datasets.mnist
(train_image, train_label), (test_image, test_label) = mnist.load_data();

model = tf.keras.models.load_model('num_reader.model');
print(model.predict([train_image]));
#predictions = model.predict([test_image]);

#pred1 = np.argmax(predictions[0]);

#print(pred1);