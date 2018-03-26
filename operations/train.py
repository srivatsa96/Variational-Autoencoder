import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cPickle
import os
from tensorflow.python import debug as tf_debug


class TrainEncoderModel(object):

    def __init__(self, config, model):

        self.config = config
        self.model = model
        self.train_writer = tf.summary.FileWriter(self.config.train_logdir)

        # Load  the CIFAR Data
        file_list = self._get_file_list(self.config.dataset_base_dir)

        all_images = []
        for file in file_list:
            image_dict = {}
            with open(file, 'rb') as f:
                print file
                image_dict = cPickle.load(f)
            all_images.append(image_dict['data'])


        self.images_dataset = np.array(all_images).reshape(-1,32,32,3)
        self.images_dataset = self.images_dataset/255.0

        self.iterator_initializer = self._setup_and_build_model()
        self.training_op = self._train_op()
        self.saver = tf.train.Saver()
        self.min_loss = None
        self.counter_train = 0




    def _get_file_list(self, base_dir):
        """
        Lists all the files inside the base directory
        """
        file_list  = os.listdir(base_dir)[1:6]
        file_list = map(lambda current_file: os.path.join(base_dir, current_file), file_list)
        return file_list

    def _train_op(self):
        """
        Setup training operation
        """
        optimizer = tf.train.AdamOptimizer(
            self.config.learning_rate).minimize(self.model.total_loss)
        return optimizer

    def _setup_and_build_model(self):
        """
        Add input graph to the overall model and build the entire graph.
        """

        images = tf.placeholder(name="images",dtype=tf.float32,shape=self.images_dataset.shape)
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.config.batch_size)
        iterator = dataset.make_initializable_iterator()

        image_batch = iterator.get_next()
        self.model.feed_input(image_batch)
        self.model.build_model()

        return iterator.initializer

    def _train_step(self, i, sess):
        """
        Running a single epoch
        """
        print('Training: ')

        feed_dict = {
            "images:0": self.images_dataset
        }
        sess.run(self.iterator_initializer, feed_dict=feed_dict)
        loop_length = self.config.dataset_size / self.config.batch_size
        net_loss = 0
        for k in tqdm(range(loop_length)):
            try:
                _, merged_summary, total_loss = sess.run(
                    [self.training_op, self.model.merged_summary, self.model.total_loss])
                net_loss += total_loss
            except tf.errors.OutOfRangeError:
                break
            self.train_writer.add_summary(merged_summary, self.counter_train)
            self.counter_train += 1
        net_loss /= loop_length

        if self.min_loss > net_loss or self.min_loss is None:
            self.min_loss = net_loss
            print('Saving Model')
            save_path = self.saver.save(sess, self.config.model_path)
        print("Total loss after %dth iteration is %f. " % (i, net_loss))

    def train_model(self,restore_model = False):
        """
        Train the entire model
        """
        print('Running training operation')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if restore_model:
                print('Restoring model: ')
                self.saver.restore(sess, self.config.model_path)
            self.train_writer.add_graph(sess.graph)
            for i in range(self.config.number_of_epoch):
                print("Running epoch %d" % (i + 1))
                self._train_step(i + 1, sess)
