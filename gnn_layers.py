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

"""
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? 
In International Conference on Learning Representations, 2019
"""
class GINLayer(tf.keras.layers.Layer):
    # input parameter : dimension of the output embedding
    def __init__(self, n_channels, activation=tf.keras.activations.relu, fixed_eps = None):
        super().__init__()
        self.n_channels = n_channels
        self.activation = activation
        self.fixed_eps = fixed_eps # GIN's epsilon can be trainable, or get a fixed value
    
    # weights dimension : [input embedding dimension, target embedding dimension]
    def build(self, inputs_shape):
        dim = inputs_shape[1][-1]
        self.W = self.add_weight(shape=(dim, self.n_channels))
        if self.fixed_eps is not None :
            self.eps = self.fixed_eps
        else :
            self.eps = self.add_weight(shape=(1,), initializer="zeros") # trainable parameter epsilon, a scalar

    
    # input : A (adjacency matrix of the graph) [batch_size, n_nodes, n_nodes]
    # input : H (nodes embeddings) [batch_size, n_nodes, input embedding dimension]
    def call(self, inputs):
        A, H = inputs[0], inputs[1]

        x = ((1 + self.eps)* H + A@H)@self.W

        if self.activation is not None :
            x = self.activation(x)
        return x

"""
M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” 
in NIPS2017
"""
class ChebLayer(tf.keras.layers.Layer):
    # input parameter : dimension of the output embedding, Order K of Chebyshev polynomial
    def __init__(self, n_channels, activation=tf.keras.activations.relu, K=3): 
        super().__init__()
        self.n_channels = n_channels
        self.activation = activation
        self.K = K
    
    # weights are initialized in the build method
    # weights dimension : [input embedding dimension, target embedding dimension]
    def build(self, inputs_shape):
        dim = inputs_shape[1][-1]
        self.W1 = self.add_weight(shape=(dim, self.n_channels))
        self.W2 = self.add_weight(shape=(dim, self.n_channels))
        self.W3 = self.add_weight(shape=(dim, self.n_channels))

        #TODO : add K weights

    
    # input : A (adjacency matrix of the graph) [batch_size, n_nodes, n_nodes]
    # input : H (nodes embeddings) [batch_size, n_nodes, input embedding dimension]
    # input : lambda_max : maximum eigenvalue of the graph Laplacian
    def call(self, inputs):
        A, H, lambda_max = inputs[0], inputs[1], inputs[2]
        # degree matrix
        D = tf.math.pow(tf.math.reduce_sum(A, axis=0), -1/2)
        # normalized Laplacian
        L = tf.eye(A.shape[0]) - D@A@D
        
        I = tf.eye(A.shape[1])
        C1 = I
        C2 = 2*L/lambda_max - I
        C3 = 2*C2@C2 - C1
        
        #TODO : implement dynamic K Chebyshev polynomial

        x = C1 @ H @ self.W1 + C2 @ H @ self.W2 + C3 @ H @ self.W3
        
        if self.activation is not None :
            x = self.activation(x)
        return x