import tensorflow as tf

# Keras recommendations on Layers subclassing : https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class SimpleGNNLayer(tf.keras.layers.Layer):
    # input parameter : dimension of the output embedding
    def __init__(self, n_channels, activation=tf.keras.activations.relu):
        super().__init__()
        self.n_channels = n_channels
        self.activation = activation
    
    # weights are initialized in the build method
    # weights dimension : [input embedding dimension, target embedding dimension]
    def build(self, inputs_shape):
        dim = inputs_shape[1][-1]
        self.W = self.add_weight(shape=(dim, self.n_channels))
    
    # input : A (adjacency matrix of the graph) [batch_size, n_nodes, n_nodes]
    # input : H (nodes embeddings) [batch_size, n_nodes, input embedding dimension]
    def call(self, inputs):
        A, H = inputs[0], inputs[1]
        x = A @ H @ self.W
        
        if self.activation is not None :
            x = self.activation(x)
        return x

class VanillaGNNLayer(tf.keras.layers.Layer):
    # input parameter : dimension of the output embedding
    def __init__(self, n_channels, activation=tf.keras.activations.relu):
        super().__init__()
        self.n_channels = n_channels
        self.activation = activation
    
    # weights are initialized in the build method
    # weights dimension : [input embedding dimension, target embedding dimension]
    def build(self, inputs_shape):
        dim = inputs_shape[1][-1]
        self.W = self.add_weight(shape=(dim, self.n_channels))
    
    # input : A (adjacency matrix of the graph) [batch_size, n_nodes, n_nodes]
    # input : H (nodes embeddings) [batch_size, n_nodes, input embedding dimension]
    def call(self, inputs):
        A, H = inputs[0], inputs[1]
        C = tf.eye(A.shape[1]) + A
        x = C @ H @ self.W
        
        if self.activation is not None :
            x = self.activation(x)
        return x

class GeneralGNNLayer(tf.keras.layers.Layer):
    # input parameter : dimension of the output embedding
    def __init__(self, n_channels, activation=tf.keras.activations.relu):
        super().__init__()
        self.n_channels = n_channels
        self.activation = activation
    
    # weights are initialized in the build method
    # weights dimension : [input embedding dimension, target embedding dimension]
    def build(self, inputs_shape):
        dim = inputs_shape[1][-1]
        self.W1 = self.add_weight(shape=(dim, self.n_channels))
        self.W2 = self.add_weight(shape=(dim, self.n_channels))
    
    # input : A (adjacency matrix of the graph) [batch_size, n_nodes, n_nodes]
    # input : H (nodes embeddings) [batch_size, n_nodes, input embedding dimension]
    def call(self, inputs):
        A, H = inputs[0], inputs[1]
        C1 = tf.eye(A.shape[1])
        x = C1 @ H @ self.W1 + A @ H @ self.W2
        
        if self.activation is not None :
            x = self.activation(x)
        return x
