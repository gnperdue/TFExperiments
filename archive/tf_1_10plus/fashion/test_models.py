import argparse
import numpy as np
import tensorflow as tf
from tffashion.keras_model_classes import ConvFashionModel

tf.enable_eager_execution()


def test_conv_model():
    model = ConvFashionModel()
    t = tf.random_normal((1, 28, 28, 1))
    out = model(t)
    assert np.any(tf.is_nan(out).numpy()) == False, "output contains nans"
    assert out.get_shape().as_list() == [1, 10], "model output shape is broken"
    assert model.count_params() == 1199882, "model parameter number is wrong"


def main(shallow):
    if shallow:
        pass
    else:
        test_conv_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shallow', default=False, action='store_true',
        help='Use shallow model'
    )
    args = parser.parse_args()
    main(**vars(args))
