import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
FLAGS.image_size = 28
FLAGS.image_color = 1
FLAGS.num_classes = 10
FLAGS.ckpt_dir = './ckpt/'


class CNNModel:

    def __init__(self, sess, name):
        # session
        self.sess = sess
        self.name = name

        # place holders
        self.x = tf.placeholder(tf.float32, [None, FLAGS.image_size * FLAGS.image_size])
        self.x_img = tf.reshape(self.x, [-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
        self.y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # build model
        self._build_model()

    def _build_model(self):

        with tf.name_scope('conv_1'):
            conv1 = tf.layers.conv2d(inputs=self.x_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)

        with tf.name_scope('conv_2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)

        with tf.name_scope('conv_3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)

        with tf.name_scope('fc_1'):
            flat = tf.reshape(pool3, [-1, 128 * 4 * 4])
            dense = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout = tf.nn.dropout(dense, self.keep_prob)

        with tf.name_scope('final_out'):
            self.logits = tf.layers.dense(inputs=dropout, units=10)

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

    def get_accuracy(self, x_test, y_test, keep_prob):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: keep_prob})


def main():

    # download training data
    tf.set_random_seed(777)  # reproducibility
    mnist = input_data.read_data_sets("MNIST_data_2/", one_hot=True)
    print("Download Done!")

    with tf.Session() as sess:
        model = CNNModel(sess, "model1")
        # restore model
        saver = tf.train.Saver()
        restore_path = FLAGS.ckpt_dir + "mnist_cnn.ckpt"
        saver.restore(sess, restore_path)

        # Test model and check accuracy
        print('Training ', mnist.train.num_examples, ' files')
        print('Testing ', mnist.test.num_examples, ' files')
        print('Accuracy:', model.get_accuracy(mnist.test.images, mnist.test.labels, 1))


main()
