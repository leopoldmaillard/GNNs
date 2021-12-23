# Graph Neural Network

This repository provides a Tensorflow 2 implementation of some GNN layers using ```tf.keras.layers.Layer``` subclassing, as well a notebook to evaluate their expressive power through an image filtering task.

## Spatial Graph Convolution

The general framework for spatial graph convolution (message passing) is :

![](https://latex.codecogs.com/svg.latex?H^{(l&plus;1)}&space;=&space;\sigma&space;(\sum_{s}{}C^{(s)}H^{(l)}W^{(l,s)}))

A GNN layer can be defined by different C matrices. For now, this project contains :

- Simple GNN : <img src="https://render.githubusercontent.com/render/math?math=C = A">
- Vanilla GNN : <img src="https://render.githubusercontent.com/render/math?math=C = I %2B A">
- General GNN : <img src="https://render.githubusercontent.com/render/math?math=C^{(1)} = I, C^{(2)} = A">
- GIN

## References

Muhammet Balcilar, Guillaume Renton, Pierre Héroux, Benoit Gaüzère, Sébastien Adam, Paul Honeine. *Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective*. ICLR 2021.
