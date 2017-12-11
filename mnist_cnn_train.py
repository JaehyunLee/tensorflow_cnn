import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
FLAGS.image_size = 28
FLAGS.image_color = 1
FLAGS.maxpool_filter_size = 2
FLAGS.batch_size = 100
FLAGS.learning_rate = 0.001
FLAGS.num_classes = 10
FLAGS.training_epochs = 15

# convolutional network layer 1
def conv1(input_data):
    FLAGS.conv1_filter_size = 3
    FLAGS.conv1_layer_size = 32
    FLAGS.stride1 = 1

    with tf.name_scope('conv_1'):
        W_conv1 = tf.Variable(tf.truncated_normal(
            [FLAGS.conv1_filter_size, FLAGS.conv1_filter_size, FLAGS.image_color, FLAGS.conv1_layer_size],
            stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([FLAGS.conv1_layer_size], stddev=0.1))
        h_conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1_relu = tf.nn.relu(tf.add(h_conv1, b1))
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv1_maxpool

# convolutional network layer 2
def conv2(input_data):
    FLAGS.conv2_filter_size = 3
    FLAGS.conv2_layer_size = 64
    FLAGS.stride2 = 1

    with tf.name_scope('conv_2'):
        W_conv2 = tf.Variable(tf.truncated_normal(
            [FLAGS.conv2_filter_size, FLAGS.conv2_filter_size, FLAGS.conv1_layer_size, FLAGS.conv2_layer_size],
            stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([FLAGS.conv2_layer_size], stddev=0.1))
        h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2_relu = tf.nn.relu(tf.add(h_conv2, b2))
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv2_maxpool

# convolutional network layer 3
def conv3(input_data):
    FLAGS.conv3_filter_size = 3
    FLAGS.conv3_layer_size = 128
    FLAGS.stride3 = 1

    with tf.name_scope('conv_3'):
        W_conv3 = tf.Variable(tf.truncated_normal(
            [FLAGS.conv3_filter_size, FLAGS.conv3_filter_size, FLAGS.conv2_layer_size, FLAGS.conv3_layer_size],
            stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal([FLAGS.conv3_layer_size], stddev=0.1))
        h_conv3 = tf.nn.conv2d(input_data, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3_relu = tf.nn.relu(tf.add(h_conv3, b3))
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv3_maxpool

# fully connected layer 1
def fc1(input_data):
    input_layer_size = 4 * 4 * FLAGS.conv3_layer_size
    FLAGS.fc1_layer_size = 625

    with tf.name_scope('fc_1'):
        input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
        W_fc1 = tf.Variable(tf.truncated_normal([input_layer_size, FLAGS.fc1_layer_size], stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size], stddev=0.1))
        h_fc1 = tf.add(tf.matmul(input_data_reshape, W_fc1), b_fc1)
        h_fc1_relu = tf.nn.relu(h_fc1)

    return h_fc1_relu

# final layer
def final_out(input_data):
    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size, FLAGS.num_classes], stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal([FLAGS.num_classes], stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)

    return h_fo

# build cnn_graph
def build_model(images, keep_prob):
    # input shape will be (*,28,28,32)
    r_cnn1 = conv1(images)
    # input shape will be (*,14,14,64)
    r_cnn2 = conv2(r_cnn1)
    # input shape will be (*,7,7,128)
    r_cnn3 = conv3(r_cnn2)
    # fully connected layer 1
    r_fc1 = fc1(r_cnn3)
    # drop out
    r_dropout = tf.nn.dropout(r_fc1, keep_prob)
    # final layer
    r_out = final_out(r_dropout)

    return r_out

def main():
    # download training data
    tf.set_random_seed(777)  # reproducibility
    mnist = input_data.read_data_sets("MNIST_data_2/", one_hot=True)
    print("Download Done!")

    # input place holders
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size * FLAGS.image_size])
    x_img = tf.reshape(x, [-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # dropout ratio
    keep_prob = tf.placeholder(tf.float32)

    # build model
    logits = build_model(x_img, keep_prob)

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning started. It takes sometime.')
    print('Training ', mnist.train.num_examples, ' files')
    for epoch in range(FLAGS.training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Learning Finished!')

    # save model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./tmp/mnist_cnn.ckpt")
    print("Model saved in file: ", save_path)

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Testing ', mnist.test.num_examples, ' files')
    print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

main()
