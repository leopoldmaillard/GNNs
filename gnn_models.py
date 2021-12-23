import tensorflow as tf
from gnn_layers import SimpleGNNLayer, VanillaGNNLayer, GeneralGNNLayer, GINLayer, ChebLayer

"""
This codes provides 3 simple GNN models that will be used to perform graph filtering.
"""

class SimpleGNNFilter(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gnn1 = SimpleGNNLayer(64)
        self.gnn2 = SimpleGNNLayer(32)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        
        x = self.gnn1(inputs)
        x = self.gnn2([inputs[0], x])
        x = self.dense(x)
        
        return x

class VanillaGNNFilter(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(64)
        self.gnn2 = VanillaGNNLayer(32)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        
        x = self.gnn1(inputs)
        x = self.gnn2([inputs[0], x])
        x = self.dense(x)
        
        return x

class GeneralGNNFilter(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gnn1 = GeneralGNNLayer(64)
        self.gnn2 = GeneralGNNLayer(32)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        
        x = self.gnn1(inputs)
        x = self.gnn2([inputs[0], x])
        x = self.dense(x)
        
        return x

class GINFilter(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gnn1 = GINLayer(64, fixed_eps=-4.0)
        self.gnn2 = GINLayer(32, fixed_eps=-3.0)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        
        x = self.gnn1(inputs)
        x = self.gnn2([inputs[0], x])
        x = self.dense(x)
        
        return x

class ChebFilter(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gnn1 = ChebLayer(32)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        
        x = self.gnn1(inputs)
        x = self.dense(x)
        
        return x