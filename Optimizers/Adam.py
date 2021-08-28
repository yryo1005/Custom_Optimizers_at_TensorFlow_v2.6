# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:16:20 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class Adam(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7,name = "Adam", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("epsilon", epsilon)
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'velocity')
            self.add_slot(var, 'accum')
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        epsilon = self._get_hyper("epsilon", var_dtype)
        
        velocity = self.get_slot(var, "velocity")
        accum = self.get_slot(var, "accum")
        velocity_t = velocity.assign( beta_1 * velocity + (1 - beta_1) * grad )
        accum_t = accum.assign( beta_2 * accum + (1 - beta_2) * grad ** 2 )
        om = velocity_t / (1 - beta_1)
        ov = accum_t / (1 - beta_2)
        var.assign( var - lr * om / (ov ** (1 / 2) + epsilon) )
        
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "beta_1" : self._serialize_hyperparameter("beta_1"),
            "beta_2" : self._serialize_hyperparameter("beta_2"),
            "epsilon" : self._serialize_hyperparameter("epsilon"),
        }
