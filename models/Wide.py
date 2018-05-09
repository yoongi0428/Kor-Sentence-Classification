# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Wide_CNN():
    def __init__(self, config, conv_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.char_size = config.charsize
        self.conv_layers = conv_layers
        self.lr = config.lr

        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.int64, shape=[None], name="output_x")
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('char_embedding', [self.char_size, self.embedding], initializer=init)

        # EMBEDDING LAYERS
        with tf.name_scope("Embedding-Layer"):
            # CNN (tf.conv2d) : input = [batch, in_height, in_width, in_channels]
            # NLP embedding only has 1 channels
            # shape = (Batch, input_len, Embedding, 1)
            if self.embedding == 0:
                # one hot encoding of zero vector, 0, 1, 2, ... , char_size - 1
                one_hot = tf.concat(
                    [
                        tf.zeros([1, self.char_size]),
                        tf.one_hot(list(range(self.char_size)), self.char_size, 1.0, 0.0)
                    ],
                    0
                )
                cnn_x = tf.nn.embedding_lookup(one_hot, self.x1)
            else:
                cnn_x = tf.nn.embedding_lookup(embedding_matrix, self.x1)
            cnn_x = tf.expand_dims(cnn_x, -1)

        cnn_output = None
        # WIDE CONVOLUTION LAYERS
        for i, conv_info in enumerate(self.conv_layers):
            # conv_info = [# of feature, kernel height, pool height]

            with tf.name_scope("Conv-Layer-" + str(i)):
                filter_width = cnn_x.get_shape()[2].value
                filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]  # [각 filter 크기, emb 크기, 1, kernel 크기]

                init = tf.contrib.layers.xavier_initializer(uniform=False)
                W = tf.get_variable("Conv_W" + str(i), filter_shape, initializer=init)
                b = tf.get_variable("Conv_b" + str(i), [conv_info[0]], initializer=init)

                conv = tf.nn.conv2d(cnn_x, W, [1, 1, 1, 1], "VALID", name="conv")

            with tf.name_scope("Non-Linear"):
                conv = tf.nn.bias_add(conv, b)
                conv = tf.nn.relu(conv)
            if conv_info[-1] != -1:
                with tf.name_scope("Max-Polling"):
                    pool_shape = [1, conv_info[-1], 1, 1]
                    conv = tf.nn.max_pool(conv, ksize=pool_shape, strides=pool_shape, padding="VALID")
            with tf.name_scope("One-Max-Pooling"):
                conv = tf.reduce_max(conv, reduction_indices=[1], keepdims=True)  # 1-max pooling
                conv = tf.squeeze(conv, [1, 2])
                if i == 0:
                    cnn_output = conv
                else:
                    cnn_output = tf.concat([cnn_output, conv], 1)

        # cnn_output : [Batch, concatenated_seq, Filter Num]
        d = cnn_output.shape[1].value
        with tf.name_scope("Output-Layer"):

            init = tf.contrib.layers.xavier_initializer(uniform=False)
            W = tf.get_variable("Output_W", [d, self.output_dim], initializer=init)
            b = tf.get_variable("Output_b", [self.output_dim], initializer=init)

            output = tf.nn.xw_plus_b(cnn_output, W, b, name="output_prob")
            self.prediction = tf.argmax(output, axis=1)

        with tf.name_scope("Loss"):
            one_hot_label = tf.one_hot(self.y_, depth=self.output_dim)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=tf.stop_gradient(one_hot_label))
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.prediction, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(config.lr, global_step, 3 * config.num_batch, 0.5, True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)

    def __str__(self):
        name = "Wide CNN"
        embeddings = "Embedding Size : " + str(self.embedding)
        filter_num = "Number of Filters : " + str(self.config.filter_num)
        layers = "Conv Layers : " + str(self.conv_layers)

        return '\n'.join([name, embeddings, filter_num, layers])