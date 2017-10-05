"""
following: https://www.tensorflow.org/tutorials/wide
and: https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/learn/wide_n_deep_tutorial.py
data: https://archive.ics.uci.edu/ml/datasets/Census+Income
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf
from tensorflow import feature_column as feature_column
from tensorflow import estimator as estimator


CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

gender = feature_column.categorical_column_with_vocabulary_list(
    'gender', ['Female', 'Male']
)
education = feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ]
)
marital_status = feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ]
)
relationship = feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ]
)
workclass = feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ]
)

# show an example of hashing
occupation = feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000
)
native_country = feature_column.categorical_column_with_hash_bucket(
    'native_country', hash_bucket_size=1000
)

# continuous base columns
age = feature_column.numeric_column('age')
education_num = feature_column.numeric_column('education_num')
capital_gain = feature_column.numeric_column('capital_gain')
capital_loss = feature_column.numeric_column('capital_loss')
hours_per_week = feature_column.numeric_column('hours_per_week')

# transformations
age_buckets = feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
)

# wide columns and deep columns
base_columns = [
    gender, education, marital_status, relationship, workclass,
    occupation, native_country, age_buckets
]

crossed_columns = [
    feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000
    ),
    feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
    ),
    feature_column.crossed_column(
        ['native_country', 'occupation'], hash_bucket_size=1000
    )
]

deep_columns = [
    feature_column.indicator_column(workclass),
    feature_column.indicator_column(education),
    feature_column.indicator_column(gender),
    feature_column.indicator_column(relationship),
    feature_column.embedding_column(native_country, dimension=8),
    feature_column.embedding_column(occupation, dimension=8),
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
]


def maybe_download(train_data, test_data):
    """maybe downloads train data and returns file names"""
    if train_data:
        train_file_name = train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            train_file.name
        )
        train_file_name = train_file.name
        train_file.close()
        print('training data is downloaded to %s' % train_file_name)

    if test_data:
        test_file_name = test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
            test_file.name
        )
        test_file_name = test_file.name
        test_file.close()
        print('test data is downloaded to %s' % test_file_name)

    return train_file_name, test_file_name


def build_estimator(model_dir, model_type):
    """build an estimator"""
    if model_type == 'wide':
        m = estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=base_columns+crossed_columns
        )
    elif model_type == 'deep':
        m = estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50]
        )
    else:
        m = estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50]
        )
    return m


def input_fn(data_file, num_epochs, shuffle):
    """input builder fn - convert data into tensors"""
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine='python',
        skiprows=1
    )
    # remove NaN elements
    df_data = df_data.dropna(how='any', axis=0)
    labels = df_data['income_bracket'].apply(lambda x: '>50k' in x).astype(int)
    return estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5
    )


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """train and evaluate the model"""
    train_file_name, test_file_name = maybe_download(train_data, test_data)
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, model_type)
    # set num_epochs to None to get infinite stream of data
    m.train(
        input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
        steps=train_steps
    )
    # set steps to None to run evaluation until all data is consumed
    results = m.evaluate(
        input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
        steps=None
    )
    print('model directory = %s' % model_dir)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir,
                   FLAGS.model_type,
                   FLAGS.train_steps,
                   FLAGS.train_data,
                   FLAGS.test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
