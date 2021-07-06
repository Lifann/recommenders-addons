# A [dynamic_embedding]() demo based on [amazon_us_reviews/Digital_Video_Games_v1_00](https://www.tensorflow.org/datasets/catalog/amazon_us_reviews)

We use [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) to build a model to predict a whether if the digital video games are purchased verifiedly.

In the demo, we expect to show how to use [dynamic_embedding.Variable]() to represent and carry embedding layers, train the model with growth of `Variable`, and restrict the `Variable` when it is growing too large.


## Start training and export model:
sh train.sh

## Stop training:
sh stop.sh

## Run inference:
sh infer.sh
