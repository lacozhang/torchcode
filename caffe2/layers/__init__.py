from . import embedding
from caffe2.python.layers import layers

layers.register_layer(embedding.Embedding.__name__, embedding.Embedding)
