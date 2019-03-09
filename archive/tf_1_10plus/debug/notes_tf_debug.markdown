Following: https://www.tensorflow.org/programmers_guide/debugger

```
(py2tf14) debug$ python -m tensorflow.python.debug.examples.debug_mnist
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting /tmp/mnist_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting /tmp/mnist_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/mnist_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/mnist_data/t10k-labels-idx1-ubyte.gz
2018-01-09 08:41:39.612691: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Accuracy at step 0: 0.1113
Accuracy at step 1: 0.2644
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
Accuracy at step 4: 0.098
Accuracy at step 5: 0.098
Accuracy at step 6: 0.098
Accuracy at step 7: 0.098
Accuracy at step 8: 0.098
Accuracy at step 9: 0.098
```

Or, with my local copy of the code...

```
(py2tf14) debug$ python debug_mnist.py
Extracting /tmp/mnist_data/train-images-idx3-ubyte.gz
Extracting /tmp/mnist_data/train-labels-idx1-ubyte.gz
Extracting /tmp/mnist_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/mnist_data/t10k-labels-idx1-ubyte.gz
2018-01-09 08:42:51.555617: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Accuracy at step 0: 0.1113
Accuracy at step 1: 0.2618
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
Accuracy at step 4: 0.098
Accuracy at step 5: 0.098
Accuracy at step 6: 0.098
Accuracy at step 7: 0.098
Accuracy at step 8: 0.098
Accuracy at step 9: 0.098
```

## Wrapping TensorFlow Sessions with `tfdbg`

```
from tensorflow.python import debug as tf_debug

...

sess = tf_debug.LocalCLIDebugWrapperSession(sess)
```

Note, Google has registered a filter, `tfdbg.has_inf_or_nan`. We may
implement our own filters (see API documentation).

## Debugging Model Training with `tfdbg`

```
python debug_mnist.py --debug
```

-> Drops us into the `tfdbg` CLI at `sess.run`...

### Frequently used commands

* Use `PageUp`, `PageDn` to move, or `fn+<up>`, `fn+<dn>`
* `run` or `r` -> takes us to the end of `Session.run()`
* `lt` -> get list of tensors -> accepts lots of flags, e.g.
    * `lt -n output/weights`
* `pt` -> print tensor
    * `pt hidden/biases/Variable`
    * Supports slicing
    * `@[coordinates]` -> navigate to a specified element in `pt` output
* `pf` -> Print value in `feed_dict` to `Session.run`
* `help` -> e.g., `help list_inputs`
* `eval`
* `ni`
* `li`
* `lo`
* `ls` - get list of sources, click to go to the file
* `ps` with path to source, print file
* `ri`
* Can navigate by clicking on `<---` or `-->` (wow!)
* `quit`

### Other features

* Redirect output with bash-style redirection.
* Lots of keyboard shortcuts

### Finding `nan`s and `inf`s


