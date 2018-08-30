from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from tffashion.data_readers import make_fashion_iterators
from tffashion.data_readers import get_data_files_dict
from tffashion.estimator_fns import shallow_model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--train-steps', default=None, type=int,
                    help='number of training steps')
parser.add_argument('--num-epochs', default=1, type=int,
                    help='number of epochs')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--tfrecord', default=False, action='store_true',
                    help='use tfrecords')
parser.add_argument('--model-dir', default='fashion', type=str,
                    help='model dir')


def predict(classifier, data_files, hyper_pars):
    # predictions is a generator - evaluation is lazy
    predictions = classifier.predict(
        input_fn=lambda: make_fashion_iterators(
            data_files['test'], hyper_pars['batch_size'],
            tfrecord=hyper_pars['tfrecord']
        ),
    )
    counter = 0
    for p in predictions:
        # TODO? - add persistency mechanism for predictions
        print(p)
        counter += 1
        if counter > 10:
            break


def evaluate(classifier, data_files, hyper_pars):
    eval_result = classifier.evaluate(
        input_fn=lambda: make_fashion_iterators(
            data_files['test'], hyper_pars['batch_size'],
            tfrecord=hyper_pars['tfrecord']
        ),
        steps=100,
    )
    print('\nEval:')
    print('acc: {accuracy:0.3f}, loss: {loss:0.3f}, MPCA {mpca:0.3f}'.format(
        **eval_result
    ))


def train_one_epoch(classifier, data_files, hyper_pars):
    classifier.train(
        input_fn=lambda: make_fashion_iterators(
            data_files['train'], hyper_pars['batch_size'],
            shuffle=True, tfrecord=hyper_pars['tfrecord']
        ),
        steps=hyper_pars['train_steps']
    )


def train(classifier, data_files, hyper_pars):
    for i in range(hyper_pars['num_epochs']):
        print('training epoch {}'.format(i))
        train_one_epoch(classifier, data_files, hyper_pars)
        print('evaluation after epoch {}'.format(i))
        evaluate(classifier, data_files, hyper_pars)


def main(
    batch_size, train_steps, num_epochs, data_dir, tfrecord, model_dir
):

    data_files = get_data_files_dict(path=data_dir, tfrecord=tfrecord)
    hyper_pars = {}
    hyper_pars['batch_size'] = batch_size
    hyper_pars['num_epochs'] = num_epochs
    hyper_pars['train_steps'] = train_steps
    hyper_pars['tfrecord'] = tfrecord

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_summary_steps=10,
        keep_checkpoint_max=3,
        model_dir=model_dir,
        tf_random_seed=None,
    )
    classifier = tf.estimator.Estimator(
        model_fn=shallow_model_fn,
        params={},
        config=run_config
    )
    train(classifier, data_files, hyper_pars)
    predict(classifier, data_files, hyper_pars)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
