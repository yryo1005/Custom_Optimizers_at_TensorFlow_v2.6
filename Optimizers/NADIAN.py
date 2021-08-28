# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:19:51 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class NADIAN(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, alpha = 0.5, beta = 0.1, momentum = 0.9, name = "NADIAN", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("alpha", alpha)
        self._set_hyper("beta", beta)
        self._set_hyper("momentum", momentum)
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'z')
            self.add_slot(var, 'pgrad')
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        alpha = self._get_hyper("alpha", var_dtype)
        beta = self._get_hyper("beta", var_dtype)
        mu = self._get_hyper("momentum", var_dtype)
        
        z = self.get_slot(var, "z")
        pgrad = self.get_slot(var, "pgrad")
        
        n_grad = (1 + mu) * grad - mu * pgrad
        var.assign( var + lr * ((1 / beta - alpha) * var - z / beta - beta * n_grad) )
        z.assign( z + lr * ((1 / beta - alpha) * var - z / beta) )
        pgrad.assign( grad )
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "alpha" : self._serialize_hyperparameter("alpha"),
            "beta" : self._serialize_hyperparameter("beta"),
            "momentum" : self._serialize_hyperparameter("momentum"),
        }