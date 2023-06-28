import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds

mnist = tf.keras.datasets.mnist
(train_image, train_label), (test_image, test_label) = mnist.load_data();

train_image = tf.keras.utils.normalize(train_image, axis=1);
test_image = tf.keras.utils.normalize(test_image, axis=1);

model = tf.keras.models.Sequential();
model.add(tf.keras.layers.Flatten());
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu));
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu));
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax));

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    );

model.fit(train_image, train_label, epochs=3);

val_loss, val_acc = model.evaluate(test_image, test_label)
print(val_loss, val_acc);
#model.save('num_reader.model');
predictions = model.predict(test_image);
pred1 = np.argmax(predictions[0]);
print(pred1);
plt.imshow(test_image[0]);
plt.show();
#pred1 = np.argmax(predictions[0]);
#print(pred1);