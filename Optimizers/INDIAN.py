# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:18:28 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class INDIAN(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, alpha = 0.5, beta = 0.1, name = "INDIAN", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("alpha", alpha)
        self._set_hyper("beta", beta)
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'z')
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        alpha = self._get_hyper("alpha", var_dtype)
        beta = self._get_hyper("beta", var_dtype)
        
        z = self.get_slot(var, "z")
        
        var.assign( var + lr * ((1 / beta - alpha) * var - z / beta - beta * grad) )
        z.assign( z + lr * ((1 / beta - alpha) * var - z / beta) )
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "alpha" : self._serialize_hyperparameter("alpha"),
            "beta" : self._serialize_hyperparameter("beta"),
        }