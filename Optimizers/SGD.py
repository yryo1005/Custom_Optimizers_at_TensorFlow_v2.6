# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:05:03 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class SGD(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, name = "SGD", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        
        var.assign( var - lr * grad )
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
        }