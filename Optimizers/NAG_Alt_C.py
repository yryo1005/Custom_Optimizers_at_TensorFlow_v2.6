# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 02:43:32 2021

@author: yryo1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib
import functools
import numpy as np
import six
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training import training_ops
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend_config
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils

class NAG_Alt_C(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, momentum = 0.99, name = "NAG_Alt_B", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", kwargs.get("mu", momentum))
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'velocity')
            # ネステロフ勾配にする前のパラメータを保持するスロット
            self.add_slot(var, "tmp_var")

    # このメソッドが1エポック毎に呼び出される
    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        # var(パラメータ)をネステロフ勾配の形に変換するメソッド
        self.set_param_as_nesterov(var_list)
        # 勾配を計算するメソッド
        grads_and_vars = self._compute_gradients(loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
        # パラメータを更新するメソッド
        return self.apply_gradients(grads_and_vars)
    
    """------------------------------------------------------------------------------------------"""    
    # 個々のvar(パラメータ)をネステロフ勾配の形に変換するメソッド
    @tf.function
    def set_param_as_nesterov(self, var_list):
        self._create_all_weights(var_list)
        mu = self._get_hyper("momentum")

        for var in var_list:
            velocity = self.get_slot(var, "velocity")
            tmp_var = self.get_slot(var, "tmp_var")
            # tmp_varに現在の値を保持しておく
            tmp_var.assign(var)
            # var自体はネステロフ勾配の形に更新
            var.assign(var + mu * velocity)

    """------------------------------------------------------------------------------------------"""
    # 勾配を計算するメソッド
    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):

        tape = tape if tape is not None else backprop.GradientTape()
        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()
        
        with tape:
            loss = self._transform_loss(loss)
        var_list = nest.flatten(var_list)
        
        with ops.name_scope_v2(self._name + "/gradients"):
            grads = tape.gradient(loss, var_list, grad_loss)
            grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None and v.dtype != dtypes.resource])

        return grads_and_vars
    
    """------------------------------------------------------------------------------------------"""
    # パラメータを更新するメソッド
    def apply_gradients(self, grads_and_vars):
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        var_list = [v for (_, v) in grads_and_vars]

        with ops.name_scope_v2(self._name):
            with ops.init_scope():
                self._create_all_weights(var_list)
            distribution = distribute_ctx.get_strategy()

            def apply_grad_to_update_var(var, grad):
                update_op = self._resource_apply_dense(grad, var)
                return update_op
            
            eagerly_outside_functions = ops.executing_eagerly_outside_functions()
            update_ops = []
            with name_scope_only_in_function_or_graph(self._name):
                for grad, var in grads_and_vars:
                    with distribution.extended.colocate_vars_with(var):
                        with name_scope_only_in_function_or_graph("update" if eagerly_outside_functions else "update_" + var.op.name):
                            update_op = distribution.extended.update(var, apply_grad_to_update_var, args=(grad,), group=False)
                            update_ops.append(update_op)
            
            with ops.control_dependencies([control_flow_ops.group(update_ops)]):
                return self._iterations.assign_add(1, read_value=False)
        
    # 個々のパラメータを更新するメソッド
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        mu = self._get_hyper("momentum", var_dtype)

        velocity = self.get_slot(var, "velocity")
        # 保持しておいた値を取得
        tmp_var = self.get_slot(var, "tmp_var")

        velocity_t = velocity.assign( mu * velocity - lr * grad )
        # 保持しておいた値を用いて勾配を更新
        var.assign( tmp_var + velocity_t )
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "momentum" : self._serialize_hyperparameter("momentum"),
        }

def name_scope_only_in_function_or_graph(name):
    if not context.executing_eagerly():
        return ops.name_scope_v1(name)
    else:
        return NullContextmanager()