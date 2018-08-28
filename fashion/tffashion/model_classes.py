import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class ShallowFashionModel(keras.Model):

    def __init__(self, num_classes=10):
        super(ShallowFashionModel, self).__init__(name='shallow_fashion_model')
        self.num_classes = num_classes
        # define layers - `input_shape` on first layer?
        self.reshape = Reshape(target_shape=[28 * 28], input_shape=(28, 28, 1))
        self.dense_1 = Dense(128, activation='relu')
        self.dense_2 = Dense(num_classes)

    def call(self, inputs):
        # define forward pass using layers defined in `__init__`
        x = self.reshape(inputs)
        x = self.dense_1(x)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # we must override this function if we want to use the subclass
        # model as part of a functional-style model - otherwise it is
        # optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


class ConvFashionModel(keras.Model):

    def __init__(self, num_classes=10):
        super(ShallowFashionModel, self).__init__(name='shallow_fashion_model')
        self.num_classes = num_classes
        # define layers - `input_shape` on first layer?
        self.conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.max_pool1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.25)
        self.flatten = Flatten()
        self.dense_1 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.dense_2 = Dense(num_classes)

    def call(self, inputs):
        # define forward pass using layers defined in `__init__`
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout2(x)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # we must override this function if we want to use the subclass
        # model as part of a functional-style model - otherwise it is
        # optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
