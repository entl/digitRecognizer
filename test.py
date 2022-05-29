import numpy
import tensorflow as tf
import os

model = tf.keras.models.load_model(os.path.join("digitsRecognizer","digits.model"), compile=True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# loss, accuracy = model.evaluate(x_test, y_test)
# print(f"loss is {loss}")
# print(f"Accuracy is {accuracy}")

# print(x_test[0].shape)

predict = model.predict(numpy.expand_dims(x_test[1],0))
# print(y_test[0])
print(numpy.argmax(predict))
print(predict)
