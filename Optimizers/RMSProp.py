# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:15:23 2021

@author: yryo1
"""

from tensorflow import keras
import tensorflow as tf

class RMSProp(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, rho = 0.9, epsilon = 1e-7,name = "RMSProp", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("rho", rho)
        self._set_hyper("epsilon", epsilon)
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'accum')
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        lr = self._get_hyper("learning_rate", var_dtype)
        rho = self._get_hyper("rho", var_dtype)
        epsilon = self._get_hyper("epsilon", var_dtype)
        
        accum = self.get_slot(var, "accum")
        
        accum_t = accum.assign(rho * accum + (1 - rho) * grad ** 2)
        var.assign( var - lr * grad / (accum_t ** (1 / 2) + epsilon) )
        
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("learning_rate"),
            "rho" : self._serialize_hyperparameter("rho"),
            "epsilon" : self._serialize_hyperparameter("epsilon"),
        }