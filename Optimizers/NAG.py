# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:10:51 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class NAG(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, momentum = 0.99, name = "NAG", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", kwargs.get("mu", momentum))
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'velocity')
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        mu = self._get_hyper("momentum", var_dtype)
        
        velocity = self.get_slot(var, "velocity")
        
        var_t = var + lr * velocity * velocity
        velocity_t = velocity.assign(mu * velocity + grad)
        var.assign(var_t - lr * velocity_t * (1 + mu))

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "momentum" : self._serialize_hyperparameter("momentum"),
        }