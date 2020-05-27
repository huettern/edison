from ann_visualizer.visualize import ann_viz
import tensorflow as tf

import sys

model_file = '../models/kws_model_medium_embedding_conv.h5'
# model_file = sys.argv[1]

# model.load_weights(model_file)
model = tf.keras.models.load_model(model_file)

# ann_viz(model, title="Artificial Neural network - Model Visualization")

tf.keras.utils.plot_model(
    model,
  to_file="model.png",
  show_shapes=True,
  show_layer_names=True,
  rankdir="TB",
  expand_nested=True,
  dpi=96,
)
