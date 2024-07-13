
import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers.convolutional import _Conv

try:
    sparse_conv2d_m = tf.load_op_library('/root/repos/tensorflow/bazel-bin/tensorflow/core/user_ops/sparse_conv2d.so')
except Exception as e:
    logging.warning(str(e))

dense_layers = {}
dense_weights = {}


class LookupAlignConvolution2d(_Conv):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 param_lambda=1.0,
                 sparse_th=0.01,
                 kernel_initializer=None,
                 bias_initializer=slim.init_ops.zeros_initializer(),
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        # L1 Regularizer
        kernel_regularizer = slim.l1_regularizer(scale=param_lambda)

        # initialize
        super(LookupAlignConvolution2d, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name, **kwargs)
        self.sparse_th = sparse_th
        self.kernel_pre = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        # dense kernel
        self.kernel_pre = self.add_variable(name='kernel_pre',
                                            shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            trainable=True,
                                            dtype=self.dtype)
        conv_th = tf.ones_like(self.kernel_pre) * self.sparse_th
        conv_zero = tf.zeros_like(self.kernel_pre)
        cond = tf.less(tf.abs(self.kernel_pre), conv_th)
        self.kernel = tf.where(cond, conv_zero, self.kernel_pre, name='kernel')

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})
        self.built = True


def lookupalign_conv(inputs,
                     filters,
                     kernel_size,
                     strides=(1, 1),
                     padding='valid',
                     dilation_rate=(1, 1),
                     activation=None,
                     use_bias=True,
                     kernel_initializer=None,
                     bias_initializer=slim.init_ops.zeros_initializer(),
                     param_lambda=1.0,
                     sparse_th=0.01,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     trainable=True,
                     name=None,
                     reuse=None):
    layer = LookupAlignConvolution2d(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format='channels_last',
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        param_lambda=param_lambda,
        sparse_th=sparse_th,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs), layer


def lookup_conv2d(tensor_in, num_outputs, kernel_size, stride, dict_size, padding=1,
                  param_lambda=0.3,
                  initial_sparsity=None, activation_fn=None,
                  biases_initializer=slim.init_ops.zeros_initializer()):

    if not initial_sparsity:
        initial_sparsity = 0.5
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    sparse_th = initial_sparsity / math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)
    stddev = 1./math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)

    padded = tf.pad(tensor_in, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
    pool_conv = slim.convolution2d(inputs=padded, num_outputs=dict_size, kernel_size=[1, 1], stride=1,
                                   padding='SAME',
                                   activation_fn=None,
                                   biases_initializer=None,
                                   scope='pool_conv')

    scope = tf.get_default_graph().get_name_scope()
    gen_sparse_conv = False
    if len(dense_weights.keys()) > 0:
        kernel_dense = dense_weights['%s/%s' % (scope, 'kernel_dense')]
        density = np.count_nonzero(kernel_dense) / kernel_dense.size
        if density < 0.15:
            gen_sparse_conv = True

    # activation for kernel weight
    if gen_sparse_conv:
        dense_kernel_shp = dense_weights['%s/%s' % (scope, 'kernel_shape')]
        dense_kernel_idx = dense_weights['%s/%s' % (scope, 'kernel')].indices
        dense_kernel_val = dense_weights['%s/%s' % (scope, 'kernel')].values
        dense_bias = tf.constant(dense_weights['%s/%s' % (scope, 'bias')])
        mode = 'custom_op'

        if mode == 'tf_op':
            # sparse convolution using only tensorflow's operations. -- SLOW!
            # im2col - image patche matrix
            img2col = tf.extract_image_patches(pool_conv, [1, kernel_size[0], kernel_size[1], 1], [1, stride[0], stride[1], 1], [1, 1, 1, 1], 'VALID')
            img2col = tf.transpose(img2col, [0, 3, 1, 2])
            img2col_shape = img2col.get_shape().as_list()
            img2col = tf.reshape(img2col, [img2col_shape[1], img2col_shape[2] * img2col_shape[3]])

            # sparse kernel & bias
            sparse_kernel = tf.SparseTensor(dense_kernel_idx, dense_kernel_val, dense_kernel_shp)

            # multiplication
            matmul = tf.sparse_tensor_dense_matmul(sparse_kernel, img2col)
            matmul = tf.transpose(matmul)
            matmul = tf.reshape(matmul, [1, img2col_shape[2], img2col_shape[3], dense_kernel_shp[0]])

            # bias & activation
            output = tf.nn.bias_add(matmul, dense_bias) if dense_bias is not None else matmul
            output = tf.nn.relu(output)
            return output
        elif mode == 'custom_op':
            conv = sparse_conv2d_m.sparse_conv2d(pool_conv, dense_kernel_idx, dense_kernel_val, dense_shape=dense_kernel_shp, strides=stride)
            output = tf.nn.bias_add(conv, dense_bias) if dense_bias is not None else conv
            output = tf.nn.relu(output)
            return output
        else:
            raise
    else:
        # dense convolution
        align_conv, layer = lookupalign_conv(inputs=pool_conv, filters=num_outputs, kernel_size=kernel_size,
                                             strides=(stride[0], stride[1]), padding='valid',
                                             param_lambda=param_lambda * sparse_th,
                                             sparse_th=sparse_th,
                                             activation=activation_fn,
                                             kernel_initializer=tf.random_uniform_initializer(-1 * stddev, stddev),
                                             bias_initializer=biases_initializer,
                                             name='align_conv')

        scope = tf.get_default_graph().get_name_scope()
        dense_layers[scope] = layer

        return align_conv


def extract_dense_weights(sess):
    for key in dense_layers.keys():
        layer = dense_layers[key]

        # sparse kernel
        dense_kernel = layer.kernel
        dense_kernel_shape = dense_kernel.get_shape().as_list()
        # dense_kernel = tf.reshape(dense_kernel, [dense_kernel_shape[0] * dense_kernel_shape[1] * dense_kernel_shape[2],
        #                                          dense_kernel_shape[3]])
        # dense_kernel = tf.transpose(dense_kernel)
        idx = tf.where(tf.not_equal(dense_kernel, 0))
        sparse_kernel = tf.SparseTensor(idx, tf.gather_nd(dense_kernel, idx), dense_kernel.get_shape())

        if layer.bias is not None:
            dk, k, b = sess.run([dense_kernel, sparse_kernel, layer.bias])
        else:
            dk, k = sess.run([dense_kernel, sparse_kernel])
            b = None
        dense_weights['%s/%s' % (key, 'kernel_dense')] = dk
        dense_weights['%s/%s' % (key, 'kernel')] = k
        dense_weights['%s/%s' % (key, 'kernel_shape')] = dense_kernel_shape
        dense_weights['%s/%s' % (key, 'bias')] = b

import logging
import multiprocessing
import sys
import time
import threading
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

import requests

import cv2

from tensorpack import imgaug
from tensorpack.dataflow import dataset
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.image import AugmentImageComponent
from tensorpack.dataflow.prefetch import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated
from tensorpack.dataflow.dataset.ilsvrc import ILSVRC12

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('requests').setLevel(logging.WARNING)


def get_mnist_data(is_train, image_size, batchsize):
    ds = MNISTCh('train' if is_train else 'test', shuffle=True)

    if is_train:
        augs = [
            imgaug.RandomApplyAug(imgaug.RandomResize((0.8, 1.2), (0.8, 1.2)), 0.3),
            imgaug.RandomApplyAug(imgaug.RotationAndCropValid(15), 0.5),
            imgaug.RandomApplyAug(imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01), 0.25),
            imgaug.Resize((224, 224), cv2.INTER_AREA)
        ]
        ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 128*10, multiprocessing.cpu_count())
        ds = BatchData(ds, batchsize)
        ds = PrefetchData(ds, 256, 4)
    else:
        # no augmentation, only resizing
        augs = [
            imgaug.Resize((image_size, image_size), cv2.INTER_CUBIC),
        ]
        ds = AugmentImageComponent(ds, augs)
        ds = BatchData(ds, batchsize)
        ds = PrefetchData(ds, 20, 2)
    return ds


def get_ilsvrc_data_alexnet(is_train, image_size, batchsize, directory):
    if is_train:
        if not directory.startswith('/'):
            ds = ILSVRCTTenthTrain(directory)
        else:
            ds = ILSVRC12(directory, 'train')
        augs = [
            imgaug.RandomApplyAug(imgaug.RandomResize((0.9, 1.2), (0.9, 1.2)), 0.7),
            imgaug.RandomApplyAug(imgaug.RotationAndCropValid(15), 0.7),
            imgaug.RandomApplyAug(imgaug.RandomChooseAug([
                imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
                imgaug.RandomOrderAug([
                    imgaug.BrightnessScale((0.8, 1.2), clip=False),
                    imgaug.Contrast((0.8, 1.2), clip=False),
                    # imgaug.Saturation(0.4, rgb=True),
                ]),
            ]), 0.7),
            imgaug.Flip(horiz=True),

            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.RandomCrop((224, 224)),
        ]
        ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 1000, multiprocessing.cpu_count())
        ds = BatchData(ds, batchsize)
        ds = PrefetchData(ds, 10, 4)
    else:
        if not directory.startswith('/'):
            ds = ILSVRCTenthValid(directory)
        else:
            ds = ILSVRC12(directory, 'val')
        ds = AugmentImageComponent(ds, [
            imgaug.ResizeShortestEdge(224, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ])
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count())
        ds = BatchData(ds, batchsize)

    return ds


class MNISTCh(dataset.Mnist):
    def __init__(self, is_train, shuffle):
        super().__init__(is_train, shuffle)

    def get_data(self):
        gen = super().get_data()
        try:
            while True:
                img, lb = next(gen)
                yield [img.reshape((28, 28, 1)), lb]
        except StopIteration as e:
            pass
        except Exception as e:
            logging.error(str(e))


class ILSVRCTenth(RNGDataFlow):
    def __init__(self, service_code):
        ILSVRCTenth.service_code = service_code
        self.cls_list = [x.decode('utf-8') for x in ILSVRCTenth._read_tenth('imagenet_lsvrc_synsets.txt').splitlines()]
        self.shuffle = True
        self.preload = 32 * 1

    @staticmethod
    def _tenthpath(pathurl):
        tenth_prefix = 'http://twg.kakaocdn.net/%s/imagenet/ILSVRC/2012/object_localization/ILSVRC/' % ILSVRCTenth.service_code
        url = tenth_prefix + pathurl
        return url

    @staticmethod
    def _read_tenth_batch(pathurls):
        import grequests
        urls = [grequests.get(ILSVRCTenth._tenthpath(pathurl)) for pathurl in pathurls]
        resps = grequests.map(urls)
        result_dict = {}
        for url, resp in zip(pathurls, resps):
            if not resp or resp.status_code // 100 != 2:
                continue
            result_dict[url] = resp.content
        return result_dict

    @staticmethod
    def _read_tenth(pathurl):
        url = ILSVRCTenth._tenthpath(pathurl)
        for _ in range(5):
            try:
                resp = requests.get(url)
                if resp.status_code // 100 != 2:
                    logging.warning('request failed code=%d url=%s' % (resp.status_code, url))
                    time.sleep(0.05)
                    continue
                return resp.content
            except Exception as e:
                logging.warning('request failed err=%s' % (str(e)))

        return ''

    def size(self):
        return len(self.train_list)

    def get_data(self):
        idxs = np.arange(len(self.train_list))
        if self.shuffle:
            self.rng.shuffle(idxs)

        caches = {}
        for i, k in enumerate(idxs):
            path = self.train_list[k]
            label = self.lb_list[k]

            if i % self.preload == 0:
                try:
                    caches = ILSVRCTenth._read_tenth_batch(self.train_list[idxs[i:i+self.preload]])
                except Exception as e:
                    logging.warning('tenth local cache failed, err=%s' % str(e))

            content = caches.get(path, '')
            if not content:
                content = ILSVRCTenth._read_tenth(path)

            img = cv2.imdecode(np.fromstring(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield [img, label]


class ILSVRCTTenthTrain(ILSVRCTenth):
    def __init__(self, service_code):
        super().__init__(service_code)

        # read image list - training
        self.train_list = ILSVRCTenth._read_tenth('ImageSets/CLS-LOC/train_cls.txt').splitlines()
        self.train_list = np.asarray(['Data/CLS-LOC/train/' + x.decode('utf-8').split(' ')[0] + '.JPEG' for x in self.train_list])

        self.lb_list = [self.cls_list.index(x.split('/')[3]) for x in self.train_list]

        self.shuffle = True


class ILSVRCTenthValid(ILSVRCTenth):
    def __init__(self, service_code):
        super().__init__(service_code)

        # read image list - validation
        self.train_list = ILSVRCTenth._read_tenth('ImageSets/CLS-LOC/val.txt').splitlines()
        self.train_list = np.asarray(['Data/CLS-LOC/val/' + x.decode('utf-8').split(' ')[0] + '.JPEG' for x in self.valid_list])

        synset_list = ILSVRCTenth._read_tenth('imagenet_validation_synsets.txt').splitlines()
        self.lb_list = [self.cls_list.index(x) for x in synset_list]

        self.shuffle = False


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=100):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logging.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        pass
                    except Exception as e:
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logging.exception("Exception in {}:{}".format(self.name, str(e)))
            except Exception as e:
                logging.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logging.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()


if __name__ == '__main__':
    df = get_mnist_data(is_train=True, image_size=224, batchsize=128)
    # df = get_ilsvrc_data_alexnet(is_train=True, image_size=224, batchsize=32)
    df.reset_state()
    generator = df.get_data()
    t0 = time.time()
    t = time.time()
    for i, dp in enumerate(generator):
        print(i, time.time() - t)
        t = time.time()
        if i == 100:
            break
    print(time.time() - t0)

#!/usr/bin/env python3
"""
Tests for the SparseConv2d Tensorflow operation.
"""

import unittest
import numpy as np
print('import tensorflow')
import tensorflow as tf
print('import libsparse_conv2d')
sparse_conv2d_m = tf.load_op_library('/root/repos/tensorflow/bazel-bin/tensorflow/core/user_ops/sparse_conv2d.so')


class InnerProductOpTest(unittest.TestCase):
    def test_sparse_conv2d(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape=(1, 228, 228, 3))
            conv = sparse_conv2d_m.sparse_conv2d(x, [[0, 0]], [1.0], dense_shape=[11, 11, 3, 96], strides=[4, 4])

            self.assertListEqual([1, 55, 55, 96], conv.get_shape().as_list())

    def test_sparse_conv2d_simple(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape=(1, 11, 11, 3))
            conv = sparse_conv2d_m.sparse_conv2d(x, [[0, 0]], [1.0], dense_shape=[11, 11, 3, 96], strides=[4, 4])

            inp = np.zeros((1, 11, 11, 3))
            out = sess.run(conv, feed_dict={x: inp})

            self.assertEqual(0, np.count_nonzero(out))

            inp[0][1][1][0] = 1
            out = sess.run(conv, feed_dict={x: inp})
            self.assertEqual(0, np.count_nonzero(out))

            inp[0][0][0][0] = 1
            out = sess.run(conv, feed_dict={x: inp})
            self.assertEqual(1, np.count_nonzero(out))


if __name__ == '__main__':
    unittest.main()

import sys
import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('requests').setLevel(logging.WARNING)

LOG_DIR = '/data/private/tf-lcnn-logs'


optimizers = {
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}

logstep = {
    'mnist': {
        'validation': 2000,
        'training': 1000,
    },
    'mnist224': {
        'validation': 500,
        'training': 200,
    },
    'ilsvrc2012': {
        'validation': 10000,
        'training': 1000,
    }
}


def get_dataset_sizes(dataset_name):
    if dataset_name == 'mnist':
        class_size = 10
        dataset_size = 60000
    elif dataset_name == 'mnist224':
        class_size = 10
        dataset_size = 60000
    elif dataset_name == 'ilsvrc2012':
        class_size = 1000
        dataset_size = 1200000
    else:
        raise Exception('invalid dataset: %s' % dataset_name)
    return class_size, dataset_size


def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


def dense_layer(tensor_in, layers, activation_fn=tf.nn.tanh, keep_prob=None):
    tensor_out = tensor_in
    for idx, layer in enumerate(layers):
        tensor_out = tf.contrib.layers.fully_connected(tensor_out, layer,
                                                       activation_fn=activation_fn,
                                                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                                       biases_initializer=tf.constant_initializer(0.001))
        tensor_out = tf.contrib.layers.dropout(tensor_out, keep_prob=keep_prob)

    return tensor_out


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


import os
import argparse
import json
import logging
import sys

import numpy as np
import tensorflow as tf
import time
import yaml

from data_feeder import get_ilsvrc_data_alexnet, get_mnist_data, DataFlowToQueue
from networks.alexnet import alexnet_model
from utils import optimizers, logstep, LOG_DIR, average_gradients, get_dataset_sizes

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Training using LCNN.')
    parser.add_argument('--conf', default='./confs/alexnet.yaml', help='configuration file path')
    parser.add_argument('--model-conf', default='lcnntest', help='lcnnbest, lcnn0.9, normal')
    parser.add_argument('--dataset', default='mnist224', help='mnist, mnist224, ilsvrc2012')
    parser.add_argument('--conv', default='lcnn', help='lcnn, conv')
    parser.add_argument('--path-ilsvrc2012', default='/data/public/ro/dataset/images/imagenet/ILSVRC/2012/object_localization/ILSVRC/')
    parser.add_argument('--logpath', default=LOG_DIR)
    parser.add_argument('--restore', type=str, default='')

    # arguments for multinode / multigpu
    parser.add_argument('--cluster', default=False, type=bool, help='True, if you train the model with multiple nodes')
    parser.add_argument('--cluster-conf', default='./confs/cluster_cloud_localps.yaml')
    parser.add_argument('--cluster-job', default='ps', help='ps, worker, local')
    parser.add_argument('--cluster-task', default=0, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--gpubatch', default='more', help='more, split')
    parser.add_argument('--warmup-epoch', default=10, type=int)

    args = parser.parse_args()

    # load config
    logging.info('config path : %s' % args.conf)
    with open(args.conf, 'r') as stream:
        conf = yaml.load(stream)
    model_conf = conf['model_conf'][args.model_conf]
    dataset = conf['datasets'][args.dataset]

    # load cluster
    if args.cluster:
        with open(args.cluster_conf, 'r') as stream:
            cluster_conf = yaml.load(stream)
        cluster = tf.train.ClusterSpec(cluster_conf)
        server = tf.train.Server(cluster, job_name=args.cluster_job, task_index=args.cluster_task)

        if args.cluster_job == 'ps':
            logging.info('parameter server %s %d' % (args.cluster_job, args.cluster_task))
            server.join()       # blocking call
            sys.exit(0)

        tfdevice = tf.train.replica_device_setter(worker_device='/job:{job}/task:{id}'.format(job=args.cluster_job, id=args.cluster_task),
                                                  cluster=cluster)
    else:
        tfdevice = '/gpu:0'

    # dataset
    class_size, dataset_size = get_dataset_sizes(args.dataset)

    # re-calculate iterations using number of gpu towers
    epochstep = dataset_size / dataset['batchsize']
    dataset['iteration'] = epochstep * dataset['epoch']
    dataset['lrstep'] = [int(x * epochstep) for x in dataset['lrepoch']]
    if args.gpubatch == 'more':
        dataset['iteration'] /= args.gpu
        dataset['lrstep'] = [x // args.gpu for x in dataset['lrstep']]
        logstep[args.dataset]['training'] = logstep[args.dataset]['training'] // args.gpu
        logstep[args.dataset]['validation'] = logstep[args.dataset]['validation'] // args.gpu
        batch_per_tower = dataset['batchsize']
        dataset['learningrate'] = [x * args.gpu for x in dataset['learningrate']]
    elif args.gpubatch == 'split':
        dataset['batchsize'] //= args.gpu

    if args.dataset == 'mnist':
        dataset_val = get_mnist_data('test', 24, batchsize=dataset['batchsize'])
    elif args.dataset == 'mnist224':
        dataset_val = get_mnist_data('test', 224, batchsize=dataset['batchsize'])
    elif args.dataset == 'ilsvrc2012':
        dataset_val = get_ilsvrc_data_alexnet('test', 224, batchsize=dataset['batchsize'], directory=args.path_ilsvrc2012)
    else:
        raise Exception('invalid dataset=%s' % args.dataset)

    # setting optimizer & learning rate
    lookup_sparse = tf.placeholder(tf.int32, name='lookup_sparse')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('train'):
        optimizer_type = optimizers[dataset['optimizer']]
        if isinstance(dataset['learningrate'], float):
            learning_rate = dataset['learningrate']
        else:
            learning_rate = tf.train.piecewise_constant(global_step, dataset['lrstep'], dataset['learningrate'])

        # gradual warm-up
        if args.gpubatch == 'more':
            warmup_iter = dataset_size * args.warmup_epoch / float(dataset['batchsize'] * args.gpu)
            warmup_ratio = tf.minimum((1.0 - 1.0 / args.gpu) * (tf.cast(global_step, tf.float32) / tf.constant(warmup_iter)) ** 2 + tf.constant(1.0 / args.gpu), tf.constant(1.0))
            learning_rate = warmup_ratio * learning_rate

        train_step = optimizer_type(learning_rate)

    # parse model configuration
    towers_inp = []
    towers_th = []
    towers_grad = []
    towers_acc = []
    towers_acc5 = []
    towers_loss = []
    with tf.variable_scope(tf.get_variable_scope()):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout prob
        for gpu_id in range(args.gpu):
            logging.info('creating tower for gpu-%d' % (gpu_id + 1))
            with tf.device(('/gpu:%d' % gpu_id) if not args.cluster else tf.train.replica_device_setter(worker_device='/job:{job}/task:{id}/gpu:{gpu_id}'.format(job=args.cluster_job, id=args.cluster_task, gpu_id=gpu_id), cluster=cluster)):
                with tf.name_scope('TASK%d_TOWER%d' % (args.cluster_task, gpu_id)) as scope:

                    with tf.device('/cpu:0'):
                        if args.dataset == 'mnist':
                            dataset_train = get_mnist_data('train', 24, batchsize=dataset['batchsize'])

                            x_img = tf.placeholder(tf.float32, shape=[dataset['batchsize'], 24, 24, 1])
                            y_ = tf.placeholder(tf.int64, shape=[dataset['batchsize']])
                            inp_th = DataFlowToQueue(dataset_train, [x_img, y_])
                            x_pre, y = inp_th.dequeue()
                            x = x_pre
                        elif args.dataset == 'mnist224':
                            dataset_train = get_mnist_data('train', 224, batchsize=dataset['batchsize'])

                            x_img = tf.placeholder(tf.float32, shape=[dataset['batchsize'], 224, 224, 1])
                            y_ = tf.placeholder(tf.int64, shape=[dataset['batchsize']])
                            inp_th = DataFlowToQueue(dataset_train, [x_img, y_])
                            x_pre, y = inp_th.dequeue()
                            x = x_pre
                        elif args.dataset == 'ilsvrc2012':
                            dataset_train = get_ilsvrc_data_alexnet('train', 224, batchsize=dataset['batchsize'], directory=args.path_ilsvrc2012)

                            x_img = tf.placeholder(tf.uint8, shape=[dataset['batchsize'], 224, 224, 3])
                            y_ = tf.placeholder(tf.int64, shape=[dataset['batchsize']])
                            inp_th = DataFlowToQueue(dataset_train, [x_img, y_])
                            x_pre, y = inp_th.dequeue()
                            x_pre = tf.cast(x_pre, tf.float32)
                            x = tf.subtract(x_pre, 128)
                        else:
                            raise Exception('invalid dataset: %s' % args.dataset)

                    towers_th.append(inp_th)
                    towers_inp.append((x_img, y_))

                    if conf['model'] == 'alexnet':
                        model = alexnet_model(x, class_size=class_size, convtype=args.conv, model_conf=model_conf, keep_prob=keep_prob)
                    else:
                        raise Exception('invalid model: %s' % conf['model'])

                    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=model))
                    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope))
                    loss = cross_entropy + loss_reg
                    towers_loss.append(loss)

                    grads = train_step.compute_gradients(loss)
                    towers_grad.append(grads)

                    correct_prediction = tf.equal(tf.argmax(model, 1), y)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    towers_acc.append(accuracy)

                    correct_prediction5 = tf.nn.in_top_k(model, y, k=5)
                    accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
                    towers_acc5.append(accuracy5)

                    tf.get_variable_scope().reuse_variables()
        pass

    # aggregate all gradients
    grads = average_gradients(towers_grad)
    acc1 = tf.reduce_mean(towers_acc)
    acc5 = tf.reduce_mean(towers_acc5)
    train_step = train_step.apply_gradients(grads, global_step=global_step)
    lss = tf.reduce_mean(towers_loss)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", lss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy-top1", acc1)
    tf.summary.scalar("accuracy-top5", acc5)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(train_step, variables_averages_op)

    logging.info('---- app configuration ---')
    print(json.dumps(conf))
    with open(os.path.join(args.logpath, 'train_conf.json'), 'w') as f:
        f.write(json.dumps(conf, indent=4))
    with open(os.path.join(args.logpath, 'conf.json'), 'w') as f:
        f.write(json.dumps({
            'model': conf['model'],
            'dataset': args.dataset,
            'conv': args.conv,
            'initial_sparsity': conf['model_conf'].get('initial_sparsity', []),
            'dictionary': conf['model_conf'].get('dictionary', [])
        }, indent=4))

    # prepare session
    saver = None
    if not args.cluster:
        saver = tf.train.Saver()
    is_chief = (args.cluster_task == 0)
    hooks = [tf.train.StopAtStepHook(last_step=dataset['iteration'])]

    with tf.Session(config=config) if not args.cluster else \
            tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, checkpoint_dir=args.logpath, hooks=hooks, config=config) as sess:
        logging.info('initialization')

        if not args.cluster:
            sess.run(tf.global_variables_initializer())
        else:
            logging.info('master: %s' % server.target)

        if saver and args.restore:
            saver.restore(sess, os.path.join(args.logpath, args.restore))

        # tensorboard
        file_writer = tf.summary.FileWriter('/date/private/tensorboard/', sess.graph)

        # enqueue thread
        coord = tf.train.Coordinator()
        for th in towers_th:
            th.set_coordinator(coord)
            th.start()

        i = 0
        if args.cluster:
            def stop_condition(): return sess.should_stop()
        else:
            def stop_condition(): return i >= dataset['iteration']

        logging.info('learning start')
        time_started = time.time()
        last_gs_num1 = last_gs_num2 = 0
        while not stop_condition():
            _, gs_num = sess.run([train_op, global_step], feed_dict={keep_prob: dataset['dropkeep']})

            if gs_num - last_gs_num1 >= logstep[args.dataset]['training']:
                train_loss, train_acc1, train_acc5, lr_val, summary = sess.run(
                    [lss, acc1, acc5, learning_rate, merged_summary_op],
                    feed_dict={keep_prob: dataset['dropkeep']}
                )

                # log of training loss / accuracy
                batch_per_sec = (args.gpu if args.gpubatch == 'more' else 1) * i / (time.time() - time_started)
                logging.info('epoch=%.2f step=%d(%d), %0.4f batchstep/sec lr=%f, loss=%g, accuracy(top1)=%.4g, accuracy(top5)=%.4g' % (gs_num / epochstep, gs_num, (i+1), batch_per_sec, lr_val, train_loss, train_acc1, train_acc5))
                last_gs_num1 = gs_num

                file_writer.add_summary(summary, gs_num)

            should_save = (gs_num - last_gs_num2 >= logstep[args.dataset]['validation'] or dataset['iteration'] - gs_num <= 1)

            if is_chief and should_save:
                # validation without batch processing
                MAXPAGE = 200
                if dataset['iteration'] - gs_num <= 1:
                    MAXPAGE = 100000
                total_acc1 = total_acc5 = 0
                total_cnt = 0
                dataset_val.reset_state()
                gen_val = dataset_val.get_data()
                for page in range(MAXPAGE):
                    # log of test accuracy
                    try:
                        images_test, ls = next(gen_val)
                    except StopIteration:
                        break

                    acc1_test, acc5_test = sess.run([accuracy, accuracy5], feed_dict={x_pre: images_test, y: ls, keep_prob: 1.0})
                    total_acc1 += acc1_test * len(ls)
                    total_acc5 += acc5_test * len(ls)
                    total_cnt += len(images_test)

                logging.info('validation(%d) accuracy(top1) %g accuracy(top5) %g' % (total_cnt, total_acc1 / total_cnt, total_acc5 / total_cnt))
                last_gs_num2 = gs_num

                if saver and args.logpath and not args.cluster:
                    saver.save(sess, os.path.join(args.logpath, 'model'), global_step=global_step)

                if args.conv == 'lcnn':
                    # print sparsity
                    gr = tf.get_default_graph()
                    tensors = [gr.get_tensor_by_name('TASK0_TOWER0/layer%d/align_conv/kernel:0' % (convid+1)) for convid in range(7)]
                    kernel_vals = sess.run(tensors)
                    logging.info('lcnn-densities: ' + ', '.join(['%f' % (np.count_nonzero(kernel_val) / kernel_val.size) for kernel_val in kernel_vals]))

            i += 1

        logging.info('optimization finished.')

    logging.info('app finished. %f' % (time.time() - time_started))

import tensorflow as tf

from layers.LookupConvolution2d import lookup_conv2d
from utils import flatten_convolution


def alex_conv_pool_layer(tensor_in, n_filters, kernel_size, stride, pool_size, pool_stride, param_lambda,
                         bias_initializer=tf.zeros_initializer(),
                         activation_fn=tf.nn.relu, padding='SAME', convtype='conv', dict_size=None, init_sparsity=None):
    if convtype == 'lcnn':
        conv = lookup_conv2d(tensor_in,
                             dict_size=dict_size,
                             initial_sparsity=init_sparsity,
                             param_lambda=param_lambda,
                             stride=stride,
                             num_outputs=n_filters,
                             kernel_size=kernel_size,
                             activation_fn=activation_fn,
                             biases_initializer=bias_initializer,
                             padding=2)
    else:
        conv = tf.contrib.layers.convolution2d(tensor_in,
                                               num_outputs=n_filters,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               activation_fn=activation_fn,
                                               weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                               biases_initializer=bias_initializer,
                                               padding=padding)
        conv = tf.nn.lrn(conv, bias=1.0, depth_radius=5, alpha=0.0001, beta=0.75)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_stride, padding=padding)
    return pool


def alexnet_model(x, class_size, convtype='lcnn', model_conf=None, keep_prob=0.5):
    with tf.variable_scope('layer1'):
        layer1 = alex_conv_pool_layer(x, 64, [11, 11], 4, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=tf.nn.relu,
                                      bias_initializer=tf.zeros_initializer(), padding='SAME', convtype=convtype,
                                      init_sparsity=model_conf['initial_sparsity'][0] if model_conf['initial_sparsity'] else None,
                                      dict_size=model_conf['dictionary'][0] if model_conf['dictionary'] else None,
                                      param_lambda=model_conf['lambda'])

    with tf.variable_scope('layer2'):
        layer2 = alex_conv_pool_layer(layer1, 192, [5, 5], 1, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=tf.nn.relu,
                                      bias_initializer=tf.constant_initializer(0.001), padding='SAME', convtype=convtype,
                                      init_sparsity=model_conf['initial_sparsity'][1] if model_conf['initial_sparsity'] else None,
                                      dict_size=model_conf['dictionary'][1] if model_conf['dictionary'] else None,
                                      param_lambda=model_conf['lambda'])

    if convtype == 'lcnn':
        with tf.variable_scope('layer3'):
            conv = lookup_conv2d(layer2,
                                 dict_size=model_conf['dictionary'][2],
                                 initial_sparsity=model_conf['initial_sparsity'][2],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=384,
                                 kernel_size=[3, 3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.zeros_initializer(),
                                 padding=1)
        with tf.variable_scope('layer4'):
            conv = lookup_conv2d(conv,
                                 dict_size=model_conf['dictionary'][3],
                                 initial_sparsity=model_conf['initial_sparsity'][3],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=1)
        with tf.variable_scope('layer5'):
            conv = lookup_conv2d(conv,
                                 dict_size=model_conf['dictionary'][4],
                                 initial_sparsity=model_conf['initial_sparsity'][4],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=1)
            pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
        with tf.variable_scope('layer6'):
            conv = lookup_conv2d(pool,
                                 dict_size=model_conf['dictionary'][5],
                                 initial_sparsity=model_conf['initial_sparsity'][5],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=4096,
                                 kernel_size=pool.get_shape().as_list()[1:3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=0)
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)
        with tf.variable_scope('layer7'):
            conv = lookup_conv2d(conv,
                                 dict_size=model_conf['dictionary'][6],
                                 initial_sparsity=model_conf['initial_sparsity'][6],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=4096,
                                 kernel_size=conv.get_shape().as_list()[1:3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=0)
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)
    else:
        with tf.variable_scope('layer3'):
            conv = tf.contrib.layers.convolution2d(layer2,
                                                   num_outputs=384,
                                                   kernel_size=[3, 3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.zeros_initializer(),
                                                   padding='SAME')
        with tf.variable_scope('layer4'):
            conv = tf.contrib.layers.convolution2d(conv,
                                                   num_outputs=256,
                                                   kernel_size=[3, 3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='SAME')
        with tf.variable_scope('layer5'):
            conv = tf.contrib.layers.convolution2d(conv,
                                                   num_outputs=256,
                                                   kernel_size=[3, 3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='SAME')
            pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
        with tf.variable_scope('layer6'):
            conv = tf.contrib.layers.convolution2d(pool,
                                                   num_outputs=4096,
                                                   kernel_size=pool.get_shape().as_list()[1:3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='VALID')
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)
        with tf.variable_scope('layer7'):
            conv = tf.contrib.layers.convolution2d(conv,
                                                   num_outputs=4096,
                                                   kernel_size=conv.get_shape().as_list()[1:3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='VALID')
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)

    with tf.variable_scope('layer8'):
        flatten = flatten_convolution(conv)
        output = tf.contrib.layers.fully_connected(flatten, class_size,
                                                   activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.zeros_initializer())

    return output


import argparse
import tensorflow as tf


if __name__ == '__main__':
    """
    Remove redundant extra gpu weights, redundant ops, and etcs
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='./models/alexnet/mnist/lcnn-accurate/model', help='model path')
    args = parser.parse_args()

    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(args.model + '.meta', clear_devices=True)
        loader.restore(sess, args.model)

        vvv = [x for x in tf.global_variables() if 'Adagrad' not in x.name and 'MovingAverage' not in x.name]
        saver = tf.train.Saver(vvv)
        saver.save(sess, args.model)

import sys
import os
import argparse
import cv2
import time
import yaml
import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.client import timeline

from layers.LookupConvolution2d import extract_dense_weights
from utils import get_dataset_sizes
from networks.alexnet import alexnet_model

config = tf.ConfigProto(
    allow_soft_placement=False,
    log_device_placement=False,
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Inference using LCNN.')
    parser.add_argument('--path', default='./models/alexnet/mnist/lcnn-fast/', help='configuration file path')
    # parser.add_argument('--path', default='/Users/ildoonet/Downloads/lcnn-fast/', help='configuration file path')
    parser.add_argument('--imgpath', type=str, default='./images/mnist_5.jpg')
    parser.add_argument('--benchmark', type=int, default=10)
    parser.add_argument('--save', type=bool, default=False)

    args = parser.parse_args()

    # load config
    logging.info('config path : %s' % args.path)
    with open(os.path.join(args.path, 'conf.json'), 'r') as stream:
        conf = yaml.load(stream)
    class_size, _ = get_dataset_sizes(conf['dataset'])
    model_conf = {key: conf.get(key, []) for key in ['initial_sparsity', 'dictionary', 'lambda']}

    # placeholders
    if conf['dataset'] == 'mnist':
        image_w = image_h = 24
        image_ch = cv2.IMREAD_GRAYSCALE
    elif conf['dataset'] == 'mnist224':
        image_w = image_h = 224
        image_ch = cv2.IMREAD_GRAYSCALE
    elif conf['dataset'] == 'ilsvrc2012':
        image_w = image_h = 224
        image_ch = cv2.IMREAD_COLOR
    else:
        raise Exception('invalid dataset: %s' % args.dataset)

    # read image & resize & center-crop to input size
    logging.info('load image')
    img = cv2.imread(args.imgpath, image_ch)

    r = 225.0 / min(img.shape[0], img.shape[1])
    dim = (int(img.shape[1] * r), int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    x = (img.shape[1] - 224) // 2 if img.shape[1] > 224 else 0
    y = (img.shape[0] - 224) // 2 if img.shape[0] > 224 else 0
    img = img[y:y + 224, x:x + 224]

    img = img.reshape((1, image_w, image_h, (1 if image_ch == cv2.IMREAD_GRAYSCALE else 3)))

    # prepare dense network
    logging.info('prepare network')
    g1 = tf.Graph()
    with g1.as_default() as g:
        with tf.device('/cpu:0'):
            if conf['dataset'] == 'mnist':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'mnist224':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'ilsvrc2012':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 3])
                x_img = tf.subtract(x_pre, 128)

            # create network graph
            if conf['model'] == 'alexnet':
                model = alexnet_model(x_img, class_size=class_size, convtype=conf['conv'], model_conf=model_conf, keep_prob=1.0)
            else:
                raise Exception('invalid model: %s' % conf['model'])

            softmax = tf.nn.softmax(model)

    with tf.Session(config=config, graph=g1) as sess:
        logging.info('start to restore - dense')
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.path, 'model'))

        logging.info('start inference - dense')

        # warmup
        input, m, output = sess.run([x_img, model, softmax], feed_dict={
            x_pre: img
        }, options=run_options, run_metadata=run_metadata)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        if args.save:
            with open('timeline_dense.json', 'w') as f:
                f.write(ctf)

        logging.info('network output = {}'.format(output))
        logging.info('predicted class = %d' % (np.argmax(output)))

        if conf['conv'] == 'lcnn':
            gr = tf.get_default_graph()
            tensors = [gr.get_tensor_by_name('layer%d/align_conv/kernel:0' % (convid + 1)) for convid in range(7)]
            kernel_vals = sess.run(tensors)
            logging.info('lcnn-densities: ' + ', '.join(['%.3f' % (np.count_nonzero(kernel_val) / kernel_val.size) for kernel_val in kernel_vals]))

        elapsed = 0
        for _ in range(args.benchmark):
            t = time.time()
            output = sess.run([model], feed_dict={
                x_pre: img
            })
            elapsed += time.time() - t
        logging.info('average elapsed time(dense) = %f' % (elapsed / args.benchmark))

        extract_dense_weights(sess)

    tf.reset_default_graph()

    if conf['conv'] == 'conv':
        sys.exit(0)

    g2 = tf.Graph()
    with g2.as_default() as g:
        with tf.device('/cpu:0'):
            if conf['dataset'] == 'mnist':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'mnist224':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'ilsvrc2012':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 3])
                x_img = tf.subtract(x_pre, 128)

            # create network graph
            if conf['model'] == 'alexnet':
                model = alexnet_model(x_img, class_size=class_size, convtype=conf['conv'], model_conf=model_conf,
                                      keep_prob=1.0)
            else:
                raise Exception('invalid model: %s' % conf['model'])

            softmax = tf.nn.softmax(model)

    with tf.Session(config=config, graph=g2) as sess:
        logging.info('start to restore - sparse')
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.path, 'model'))

        logging.info('start inference - sparse')

        # warmup
        output = sess.run([softmax], feed_dict={
            x_pre: img
        })
        output = sess.run([softmax], feed_dict={
            x_pre: img
        }, options=run_options, run_metadata=run_metadata)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        if args.save:
            with open('timeline_sparse.json', 'w') as f:
                f.write(ctf)

        logging.info('network output = {}'.format(output))
        logging.info('predicted class = %d' % (np.argmax(output)))

        elapsed = 0
        for _ in range(args.benchmark):
            t = time.time()
            output = sess.run([model], feed_dict={
                x_pre: img
            })
            elapsed += time.time() - t
        logging.info('average elapsed time(sparse) = %f' % (elapsed / args.benchmark))

