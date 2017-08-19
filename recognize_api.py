import numpy as np
import librosa
from .model import *
from . import data
from .data import index2str

# See https://github.com/fchollet/keras/issues/2397 for explanation on why we need to save the graph just after init
# when working with multi threads
# See https://www.tensorflow.org/api_docs/python/tf/Graph for working with several graph
graph = tf.get_default_graph()


def init():
    global graph
    with graph.as_default():
        print('init model')
        # tf.reset_default_graph()
        # g = tf.Graph()
        # with g.as_default():
        batch_size = 1  # batch size

        # vocabulary size
        voca_size = data.voca_size

        # mfcc feature of audio
        x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))

        # sequence length except zero-padding
        seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

        # encode audio feature
        logit = get_logit(x, voca_size=voca_size)

        # ctc decoding
        decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

        # to dense tensor
        y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

        # Start the session
        sess = tf.Session()  # graph=g)
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('submodule/sttw/asset/train'))

    return sess, x, y


def recognize_file(sess, x, y, file):
    # load wave file
    wav, _ = librosa.load(file, mono=True, sr=16000)
    # get mfcc feature
    mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])

    # run session
    label = sess.run(y, feed_dict={x: mfcc})

    output = ''
    for index_list in label:
        output += index2str(index_list)
    return output
