"""
Following:
https://github.com/mhyttsten/Misc/blob/master/Blog_Estimators_DataSet.py
https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
"""
from __future__ import print_function

import tensorflow as tf
import os
import sys
if sys.version_info < (3, 0, 0):
    from urllib import urlopen
else:
    from urllib.request import urlopen

tf_version = tf.__version__
print('TF verion: {}'.format(tf_version))
assert "1.3" <= tf_version, "TF 1.3 or later is required."

PATH = '/tmp/tf_dset_estmtr_apis'

# fetch and store training and test data
PATH_DATASET = PATH + os.sep + 'dataset'
FILE_TRAIN = PATH_DATASET + os.sep + 'iris_training.csv'
FILE_TEST = PATH_DATASET + os.sep + 'iris_test.csv'
URL_TRAIN = 'http://download.tensorflow.org/data/iris_training.csv'
URL_TEST = 'http://download.tensorflow.org/data/iris_test.csv'


def downloadDataset(url, filen, path=PATH_DATASET):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(filen):
        data = urlopen(url).read()
        with open(filen, 'wb') as f:
            f.write(data)


downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

feature_names = [
    'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'
]


def my_input(file_path, perform_shuffle=False, repeat_count=1):
    """
    create an input function reading a file with the Dataset API
    """
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:]
        del parsed_line[-1]
        features = parsed_line
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path).skip(1).map(decode_csv))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


next_batch = my_input(FILE_TRAIN, True)

feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

# dnn regression classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir=PATH
)

# train, stop after 8 epochs
classifier.train(
    input_fn=lambda: my_input(FILE_TRAIN, True, 8)
)

# eval, 4 epochs?
evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input(FILE_TEST, False, 4)
)
print('Evaluation results')
for key in evaluate_result:
    print('   {}, was: {}'.format(key, evaluate_result[key]))

# predict some flowers
predict_results = classifier.predict(
    input_fn=lambda: my_input(FILE_TEST, False, 1)
)
print('Predictions on test file')
for prediction in predict_results:
    print(prediction['class_ids'][0])

# create a dset for prediction
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa


def new_input():
    def decode(x):
        x = tf.split(x, 4)
        return dict(zip(feature_names, x))

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None


predict_results = classifier.predict(input_fn=new_input)

print('Predictions on memory')
for idx, prediction in enumerate(predict_results):
    type = prediction['class_ids'][0]
    if type == 0:
        print("I think: {}, is Iris Sentosa".format(prediction_input[idx]))
    elif type == 1:
        print("I think: {}, is Iris Versicolor".format(prediction_input[idx]))
    else:
        print("I think: {}, is Iris Virginica".format(prediction_input[idx]))
