# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:07:32 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class Momentum(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, momentum = 0.99, name = "Momentum", **kwargs):
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
        
        momentum = self.get_slot(var, "velocity")
        
        momentum_t = momentum.assign( mu * momentum + (1.0 - mu) * grad )
        var.assign( var - lr * momentum_t )
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "momentum" : self._serialize_hyperparameter("momentum"),
        }