import numpy as np
import tensorflow as tf


class VariationalAutoencoder(object):

    def __init__(self, config, mode='train'):

        self.config = config
        self.mode = mode  # Either train or generate

        # A [batch_size,3,image_height,image_width] input image tensor
        self.input_image = None

    def _encoder_network(self, input_image):
        """
        A Probablistic encoder that maps the input image from input space to the latent space
        Params:
        input_image : Input Image to be encoded
        Returns:
        encoded_mean
        encoded_sigma
        """
        print('Setting Encoder Network')
        with tf.name_scope('Encoder'):
            with tf.name_scope('Conv1'):
                conv1 = tf.layers.conv2d(inputs=input_image, filters=32, kernel_size=[
                                         3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            # Output Shape [32,32,32]
            with tf.name_scope('Conv2'):
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[
                                         3, 3], strides=(2, 2), padding='VALID', activation=tf.nn.relu)
            # Output Shape [15,15,64]
            with tf.name_scope('MaxPool3'):
                max_pool3 = tf.layers.max_pooling2d(
                    inputs=conv2, pool_size=[2, 2], strides=(2, 2), padding='VALID')
            # Output Shape [7,7,64]

            flatten = tf.reshape(max_pool3, shape=[-1, 7*7*64])
            # Output shape [batch_size, 7*7*64]

            with tf.name_scope('FC1'):
                fc1 = tf.layers.dense(
                    inputs=flatten, units=self.config.hidden_dim, activation=tf.nn.relu)

            with tf.name_scope('EncodedMean'):
                encoded_mean = tf.layers.dense(
                    inputs=fc1, units=self.config.latent_vector_length, activation=tf.nn.sigmoid)

            with tf.name_scope('EncodedSigma'):
                encoded_sigma = tf.layers.dense(
                    inputs=fc1, units=self.config.latent_vector_length, activation=tf.nn.sigmoid)

        # with tf.name_scope('Conv4'):
        #     conv4 = tf.layers.conv2d(inputs=max_pool3, filters=64, kernel_size=[
        #                              3, 3], strides=(1, 1), padding='SAME')
        # with tf.name_scope('Conv5'):
        #     conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[
        #                              3, 3], strides=(2, 2), padding='VALID')
        # with tf.name_scope('MaxPool6'):
        #     max_pool6 = tf.layers.max_pooling2d(inputs=conv5, filters=64, kernel_size=[
        #                                         3, 3], strides=(2, 2), padding='VALID')

        return encoded_mean, encoded_sigma

    def _decoder_network(self, latent_variable):
        """
        A Probablistic decoder network that decodes the input image from latent space to generate images
        Params:
        latent_variable: Input Vector sampled from Latent Space
        Returns:
        """
        print('Setting decoder network')
        with tf.name_scope('Decoder'):
            with tf.name_scope('DecodeLatentSpace'):
                hidden_state1 = tf.layers.dense(
                    inputs=latent_variable, units=self.config.hidden_dim, activation=tf.nn.relu)
            with tf.name_scope('DecodeFC1'):
                decode_fc1 = tf.layers.dense(
                    inputs=hidden_state1, units=7 * 7 * 64, activation=tf.nn.relu)

            deflatten = tf.reshape(decode_fc1,[-1, 7, 7, 64])

            with tf.name_scope('TransConv3'):
                decode_conv3 = tf.layers.conv2d_transpose(inputs=deflatten, filters=64, kernel_size=[3,3], strides=(
                    2, 2), padding='VALID', activation=tf.nn.relu)

            with tf.name_scope('TransConv2'):
                decode_conv2 = tf.layers.conv2d_transpose(inputs=decode_conv3,kernel_size=[4,4], filters=32, strides=(
                    2, 2), padding='VALID', activation=tf.nn.relu)

            with tf.name_scope('TransConv1'):
                reconstructed_image = tf.layers.conv2d_transpose(
                    inputs=decode_conv2, filters=3, strides=(1, 1),kernel_size=[3,3], padding='SAME', activation=tf.nn.sigmoid)

        return reconstructed_image

    def feed_input(self, image_batch):
        """
        Feed input to the model
        """
        print('Setting feed_input')
        self.input_image = image_batch

    def _setup_input(self):
        """
        Setup input for the model
        """

        self.input_image = tf.placeholder(name="input_image", dtype=tf.float32, shape=[
                                          None, self.config.image_height, self.config.image_width, 3])

    def _setup_loss(self, generated_image, mean, std_dev):
        """
        Setup loss that needs to be optimized
        """
        print('Setting loss calculation')
        with tf.variable_scope('model_loss'):
            self.input_image_flat = tf.reshape(
                self.input_image, [self.config.batch_size, -1])
            generated_image_flat = tf.reshape(
                generated_image, [self.config.batch_size, -1])
            with tf.variable_scope('generation_loss'):
                generation_loss = -tf.reduce_sum((self.input_image_flat * tf.log(1e-8 + generated_image_flat) + (
                    1 - self.input_image_flat) * tf.log((1e-8 + 1) - generated_image_flat)), 1)
            with tf.variable_scope('latent_loss'):
                latent_loss = 0.5 * \
                    tf.reduce_sum((tf.square(mean) + tf.square(std_dev) -
                                   tf.log(tf.square(std_dev)) - 1), 1)
                total_loss = tf.reduce_mean(generation_loss + latent_loss)
                tf.losses.add_loss(total_loss)
                total_loss = tf.losses.get_total_loss()
                tf.summary.scalar(self.mode + '_loss', total_loss)
            return total_loss

    def build_model(self):
        """
        Build the model
        """
        print('Building model for the %s mode operation' % (self.mode))
        if self.mode == 'train':
            mean, stddev = self._encoder_network(self.input_image)
            sample = tf.random_normal(
                [self.config.batch_size, self.config.latent_vector_length], 0, 1, dtype=tf.float32)
            guess_sample = stddev * sample + mean
            self.reconstructed_image = self._decoder_network(guess_sample)
            self.total_loss = self._setup_loss(
                self.reconstructed_image, mean, stddev)

            for variable in tf.trainable_variables():
                tf.summary.histogram(variable.name, variable)

            self.merged_summary = tf.summary.merge_all()
        else:
            self.sample_prior = tf.placeholder(
                name="sample_prior", dtype=tf.float32, shape=[self.latent_vector_length])
            self.sample_prior = tf.expand_dim(self.sample_prior, 0)
            self.reconstructed_image = self._decoder_network(self.sample_prior)
