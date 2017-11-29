# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt
import time

from tensorflow.examples.tutorials.mnist import input_data

def printLog(*message):
    now = time.localtime()
    print("%04d-%02d-%02d %02d:%02d:%02d " % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec),message)

tf.set_random_seed(777)  # reproducibility

printLog("Start")

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.name_scope('conv_01') as scope:

                # Convolutional Layer #1
#               conv1 = tf.nn.conv2d(X_img, tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)), strides=[1,1,1,1], padding='SAME')
#               conv1 = tf.nn.relu(conv1)
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)

#               Pooling Layer #1
#               pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)

#               dropout1 = tf.nn.dropout(pool1, keep_prob=0.7)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)
                self.conv1_hist = tf.summary.histogram("conv1", conv1)
                self.pool1_hist = tf.summary.histogram("pool1", pool1)
                self.dropout1_hist = tf.summary.histogram("dropout1", dropout1)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope('conv_02') as scope:
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)
                self.conv2_hist = tf.summary.histogram("conv2", conv2)
                self.pool2_hist = tf.summary.histogram("pool2", pool2)
                self.dropout2_hist = tf.summary.histogram("dropout2", dropout2)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope('conv_03') as scope:
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)
                self.conv3_hist = tf.summary.histogram("conv3", conv3)
                self.pool3_hist = tf.summary.histogram("pool3", pool3)
                self.dropout3_hist = tf.summary.histogram("dropout3", dropout3)

            # Dense Layer with Relu

            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])

#            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
#            b4 = tf.Variable(tf.random_normal([625]))
#            dense4 = tf.nn.relu(tf.matmul(flat, W4) + b4)

            with tf.name_scope('layer_01') as scope:
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
                self.dense4_hist = tf.summary.histogram("dense4", dense4)
                self.dropout4_hist = tf.summary.histogram("dropout4", dropout4)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            with tf.name_scope('layer_02') as scope:
                self.logits = tf.layers.dense(inputs=dropout4, units=10)
                self.logits_hist = tf.summary.histogram("logits", self.logits)

        # define cost/loss & optimizer
        with tf.name_scope('cost') as scope:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.cost_sum = tf.summary.scalar("cost", self.cost)

        with tf.name_scope('optimizer') as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        with tf.name_scope('accuracy') as scope:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracy_sum = tf.summary.scalar("cost", self.accuracy)

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, merged_summary, x_data, y_data, training=True):
        return self.sess.run([merged_summary, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/9.5.mnist_cnn_layer")
writer.add_graph(sess.graph)  # Show the graph

sess.run(tf.global_variables_initializer())

printLog('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, c, _ = m1.train(merged_summary, batch_xs, batch_ys)
        avg_cost += c / total_batch
        writer.add_summary(summary, global_step=epoch * total_batch + i)

    printLog('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

#    printLog(('Epoch: %04d' % (epoch + 1)) + (' cost =' '{:.9f}'.format(avg_cost)))
#    print('Accuracy:',  m1.get_accuracy(mnist.test.images, mnist.test.labels))

printLog('Learning Finished!')

# Test model and check accuracy
printLog('Final Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

'''
Learning Started!
('Epoch:', '0001', 'cost =', '0.812571620')
('Epoch:', '0002', 'cost =', '0.295275812')
('Epoch:', '0003', 'cost =', '0.232211800')
('Epoch:', '0004', 'cost =', '0.198430825')
('Epoch:', '0005', 'cost =', '0.182073716')
('Epoch:', '0006', 'cost =', '0.171181509')
('Epoch:', '0007', 'cost =', '0.164130612')
('Epoch:', '0008', 'cost =', '0.152903450')
('Epoch:', '0009', 'cost =', '0.151070237')
('Epoch:', '0010', 'cost =', '0.149221859')
('Epoch:', '0011', 'cost =', '0.145802813')
('Epoch:', '0012', 'cost =', '0.143260657')
('Epoch:', '0013', 'cost =', '0.137704694')
('Epoch:', '0014', 'cost =', '0.137128670')
('Epoch:', '0015', 'cost =', '0.134011376')
Learning Finished!
('Accuracy:', 0.98820001)
'''

'''
Learning Started!
('Epoch:', '0001', 'cost =', '0.812571620')
Learning Finished!
('Accuracy:', 0.96020001)
'''

'''
('Epoch:', '0001', 'cost =', '0.803601784', 'Accuracy:', 0.96200001)
('Epoch:', '0002', 'cost =', '0.284448627', 'Accuracy:', 0.9752)
('Epoch:', '0003', 'cost =', '0.232934248', 'Accuracy:', 0.98110002)
('Epoch:', '0004', 'cost =', '0.198133062', 'Accuracy:', 0.98329997)
('Epoch:', '0005', 'cost =', '0.184169187', 'Accuracy:', 0.98509997)
('Epoch:', '0006', 'cost =', '0.171080997', 'Accuracy:', 0.98500001)
('Epoch:', '0007', 'cost =', '0.161481809', 'Accuracy:', 0.98760003)
('Epoch:', '0008', 'cost =', '0.154639998', 'Accuracy:', 0.98879999)
('Epoch:', '0009', 'cost =', '0.146488340', 'Accuracy:', 0.98830003)
('Epoch:', '0010', 'cost =', '0.147604724', 'Accuracy:', 0.9885)
('Epoch:', '0011', 'cost =', '0.147946547', 'Accuracy:', 0.98979998)
('Epoch:', '0012', 'cost =', '0.142482386', 'Accuracy:', 0.98949999)
('Epoch:', '0013', 'cost =', '0.139422286', 'Accuracy:', 0.98989999)
('Epoch:', '0014', 'cost =', '0.139284524', 'Accuracy:', 0.9896)
('Epoch:', '0015', 'cost =', '0.133563989', 'Accuracy:', 0.9903)
Learning Finished!
('Final Accuracy:', 0.9903)
'''



