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

def name_scope_only_in_function_or_graph(name):
    if not context.executing_eagerly():
        return ops.name_scope_v1(name)
    else:
        return NullContextmanager()

class NullContextmanager(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        pass
    
    def __exit__(self, type_arg, value_arg, traceback_arg):
        return False

class ML_QN2(keras.optimizers.Optimizer):
    """ Optimizer that implements the ML_MoQ algorithm
    
    ML_QN(Memorry Less Quasi-Newton Method) solves the problem of QN(Quasi-Newton method)
    that it requires an approximate matrix of Hessian, which makes it un suitable for the training
    of large-scale, by learning without the storage of matrix.
    """
    def __init__(self, lr = 1.0, amendment = False, apply_theta = False, name = "ML_QN2", **kwargs):
        """
        Args:
            lr : larning rate. Defaults to 1.0.
            amendment : flag of amendment term of Y. If amendment equals True, amendment term of Y will work. More details to follow. Defaults to False
            apply_theta : flag of limits the range of theta. If apply_theta equals True, the range of theta will be limited. Defaults to False
        
        """
        super().__init__(name, **kwargs)
        # Parameters related main work of ML_QN
        self._set_hyper("lr", lr)
        self._set_hyper("theta", 0.0)
        self._set_hyper("sg", 0.0)
        self._set_hyper("yg", 0.0)
        self._set_hyper("sy", 0.0)
        self._set_hyper("yy", 0.0)
        
        # flags
        self.amendment = amendment
        self.apply_theta = apply_theta

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'one_past_var')
            self.add_slot(var, 'g')
            self.add_slot(var, 's')
            self.add_slot(var, 'y')
            self.add_slot(var, 'z')
            self.add_slot(var, "one_past_grad")
    
    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        grads_and_vars = self._compute_gradients(loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
        return self.apply_gradients(grads_and_vars)

    """ ------------------------------------------------------------------------"""
    # Ml_QN
    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        var_list = [v for (_, v) in grads_and_vars]

        with ops.name_scope_v2(self._name):
            with ops.init_scope():
                self._create_all_weights(var_list)
            
            if not grads_and_vars:
                return control_flow_ops.no_op()

            if distribute_ctx.in_cross_replica_context():
                raise RuntimeError("`apply_gradients() cannot be called in cross-replica context. ""Use `tf.distribute.Strategy.run` to enter replica ""context.")
            
            strategy = distribute_ctx.get_strategy()
            if (not experimental_aggregate_gradients and strategy and
                isinstance(strategy,
                           (parameter_server_strategy.ParameterServerStrategyV1,
                            parameter_server_strategy_v2.ParameterServerStrategyV2,
                            central_storage_strategy.CentralStorageStrategy,
                            central_storage_strategy.CentralStorageStrategyV1))):
                raise NotImplementedError("`experimental_aggregate_gradients=False is not supported for ""ParameterServerStrategy and CentralStorageStrategy")
            
            apply_state = self._prepare(var_list)
            if experimental_aggregate_gradients:
                grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
                grads_and_vars = self._aggregate_gradients(grads_and_vars)
            grads_and_vars = self._transform_gradients(grads_and_vars)

            if optimizer_utils.strategy_supports_no_merge_call():
                return self._distributed_apply(strategy, grads_and_vars, name, apply_state)
            
            else:
                return distribute_ctx.get_replica_context().merge_call(
                    functools.partial(self._distributed_apply, apply_state=apply_state),
                    args=(grads_and_vars,),
                    kwargs={
                        "name": name,
                    })
                
    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
        def apply_grad_to_update_var(var, grad):
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)
            
            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError("Cannot use a constraint function on a sparse variable.")
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(grad.values, var, grad.indices, **apply_kwargs)
            
            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
            if var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return var.assign(var.constraint(var))
            else:
                return update_op

        # calculate the necessary parameters such as inner product
        self.prepare_apply(grads_and_vars)

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with name_scope_only_in_function_or_graph(name or self._name):
            for grad, var in grads_and_vars:
                with distribution.extended.colocate_vars_with(var):
                    with name_scope_only_in_function_or_graph("update" if eagerly_outside_functions else "update_" + var.op.name):
                        update_op = distribution.extended.update(var, apply_grad_to_update_var, args=(grad,), group=False)
                        if distribute_ctx.in_cross_replica_context():
                            update_ops.extend(update_op)
                        else:
                            update_ops.append(update_op)
        
        any_symbolic = any(isinstance(i, ops.Operation) or tf_utils.is_symbolic_tensor(i) for i in update_ops)
        if not context.executing_eagerly() or any_symbolic:
            with backend._current_graph(update_ops).as_default():  # pylint: disable=protected-access
                with ops.control_dependencies([control_flow_ops.group(update_ops)]):
                    return self._iterations.assign_add(1, read_value=False)
        
        return self._iterations.assign_add(1)
    
    # calculate the necessary parameters such as inner product
    @tf.function
    def prepare_apply(self, grads_and_vars):
        tmp_ZS = 0.0
        tmp_SS = 0.0
        norm_g = 0.0
        tmp_SG = 0.0
        tmp_YG = 0.0
        tmp_SY = 0.0
        tmp_YY = 0.0
        
        if self.amendment:
            for grad, var in grads_and_vars:
                z = self.get_slot(var, "z")
                z_t = z.assign( grad - self.get_slot(var, "one_past_grad") )
    
                tmp_ZS += tf.reduce_sum( self.get_slot(var, "s") * z_t )
                tmp_SS += tf.reduce_sum( self.get_slot(var, "s") * self.get_slot(var, "s") )
                norm_g += tf.reduce_sum( grad * grad )
    
            w = 2.0 if norm_g > 1e-2 else 100.0
            delta = tf.maximum(tmp_ZS / tmp_SS, 0)
            xi = w * tf.math.sqrt(norm_g) + delta
    
            for grad, var in grads_and_vars:
                y = self.get_slot(var, "y")
                y_t = y.assign( self.get_slot(var, "z") + xi * self.get_slot(var, "s") )
    
                tmp_SG += tf.reduce_sum( self.get_slot(var, "s") * grad )
                tmp_YG += tf.reduce_sum( y_t * grad )
                tmp_SY += tf.reduce_sum( self.get_slot(var, "s") * y_t )
                tmp_YY += tf.reduce_sum( y_t * y_t )
        
        else:
            for grad, var in grads_and_vars:
                y = self.get_slot(var, "y")
                y_t = y.assign( grad - self.get_slot(var, "one_past_grad") )
    
                tmp_SG += tf.reduce_sum( self.get_slot(var, "s") * grad )
                tmp_YG += tf.reduce_sum( y_t * grad )
                tmp_SY += tf.reduce_sum( self.get_slot(var, "s") * y_t )
                tmp_YY += tf.reduce_sum( y_t * y_t )
        
        sg = self._get_hyper("sg")
        sg.assign( tmp_SG )
        yg = self._get_hyper("yg")
        yg.assign( tmp_YG )
        sy = self._get_hyper("sy")
        sy.assign( tmp_SY )
        yy = self._get_hyper("yy")
        yy.assign( tmp_YY )
        theta = self._get_hyper("theta")
        theta.assign( tmp_SY / tmp_YY )
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("lr")

        theta = self._get_hyper("theta")
        
        if self.apply_theta:
            if theta < 0: theta = lr
            elif theta > 1: 1

        sg = self._get_hyper("sg")
        yg = self._get_hyper("yg")
        sy = self._get_hyper("sy")
        yy = self._get_hyper("yy")

        s = self.get_slot(var, "s")
        y = self.get_slot(var, "y")

        one_past_grad = self.get_slot(var, "one_past_grad")
        one_past_var = self.get_slot(var, "one_past_var")

        if self.iterations == 0:
            direction = -1.0 * grad

            one_past_var_t = one_past_var.assign( var )
            one_past_grad.assign( grad )
            var_t = var.assign( var + lr * direction )
        
        else:
            direction = -1.0 * ( theta * grad - (theta * y * (sg / sy) + theta * s * (yg / sy)) 
                                + (1 + (theta * yy / sy)) * s * (sg / sy) )

            one_past_var_t = one_past_var.assign( var )
            one_past_grad.assign( grad )
            var_t = var.assign( var + lr * direction )

        s.assign( var_t - one_past_var_t )