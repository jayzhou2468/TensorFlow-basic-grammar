# -*- coding: utf-8 -*-
# @Time    : 2019/8/12 下午2:41
# @Author  : Ryan
# @File    : base_model.py

""""Basic Seq2Seq model with VAE, no Attention support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import collections
import tensorflow as tf

import model_helper as _mh

from utils.log import log_info as _info
from utils.log import log_error as _error


def get_scpecific_scope_params(scope=''):
	"""used to get specific parameters for training
	"""
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

class TrainOutputTuple(collections.namedtuple('TrainOutputTuple',
		'train_loss predict_count global_step batch_size learning_rate')):
	pass

class EvalOutputTuple(collections.namedtuple('EvalOutputTuple',
		'eval_loss predict_count batch_size')):
	pass

class InferOutputTuple(collections.namedtuple('InferOutputTuple',
		'infer_logits sample_id')):
	pass


class BaseModel(object):
    """Base Model"""
    def __init__(self):

        # load parameters
        self._set_params_initializer(hyparams, mode, scope)
        # build graph
        res = self.build_graph(hyparams, scope)
        # optimizer or infer
        self._train_or_inference(hyparams, res)

        # saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hyparams.num_keep_ckpts
        )
    def _set_params_initializer(self, hyparams, mode, scope):
        """load  the parameters and set the initializer"""
        self.mode = mode
        # pre_train flag is used for distinguish with pre_train and fine tune
        if hyparams.enable_vae:
            _info('Enable VAE')
            self.enable_vae = True
            self.pre_train = hyparams.pre_train
        else:
            self.enable_vae = False
            self.pre_train = False
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False)

        # define the input
        self.encoder_input_data = tf.placeholder(
            tf.int32, [None, None], name='encoder_input_data'
        )
        self.decoder_input_data = tf.placeholder(
            tf.int32, [None, None], name='decoder_input_data'
        )
        self.decoder_output_data = tf.placeholder(
            tf.int32, [None, None], name='decoder_output_data'
        )
        self.seq_length_encoder_input_data = tf.placeholder(
            tf.int32, [None], name='seq_length_encoder_input_data'
        )
        self.seq_length_decoder_input_data = tf.placeholder(
            tf.int32, [None], name='seq_length_decoder_input_data'
        )

        # load some important hypramas
        self.unit_type = hyparams.unit_type
        self.num_units = hyparams.num_units
        self.num_encoder_layers = hyparams.num_encoder_layers
        self.num_decoder_layers = hyparams.num_decoder_layers
        self.num_encoder_residual_layers = self.num_encoder_layers - 1
        self.num_decoder_residual_layers = self.num_decoder_layers - 1

        # set initializer
        random_seed = hyparams.random_seed
        initialier = _mh.get_initializer(hyparams.init.op, random_seed, hyparams.init_weight)
        tf.get_variable_scope().set_initializer(initialier)

        def init_embeddings(self, hyparams, scope):
            """init embeddings"""
            self.embedding_encoder, self.embedding_decoder = \
                _mh.create_emb_for_encoder_and_decoder(
                    share_vocab = hyparams.share_vocab,
                    src_vocab_size = self.src_vocab_size,
                    tgt_vocab_size = self.tgt_vocab_size,
                    src_embed_size = self.num_units,
                    tgt_embed_size = self.num_units,
                    scope = scope
                )

        def build_graph(self, hyparams, scope)  
















































