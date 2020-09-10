import numpy
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

import tensorflow as tf
print(tf.__version__)

x = tf.Variable( 3, name="x")
y = tf.Variable( 4, name="y")


@tf.function
def func( x, y ):
    return x*x*y + y + 2

z = func( x, y )
print( z )

