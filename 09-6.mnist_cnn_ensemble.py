# Lab 11 MNIST and Deep learning CNN
# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 20
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

            # Convolutional Layer #1
            with tf.name_scope('conv_01') as scope:
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
                # Pooling Layer #1
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],padding="SAME", strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1,rate=0.7, training=self.training)
                self.conv1_hist = tf.summary.histogram(self.name + "conv1", conv1)
                self.pool1_hist = tf.summary.histogram(self.name + "pool1", pool1)
                self.dropout1_hist = tf.summary.histogram(self.name + "dropout1", dropout1)


            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope('conv_02') as scope:
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2,rate=0.7, training=self.training)
                self.conv2_hist = tf.summary.histogram(self.name + "conv2", conv2)
                self.pool2_hist = tf.summary.histogram(self.name + "pool2", pool2)
                self.dropout2_hist = tf.summary.histogram(self.name + "dropout2", dropout2)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.name_scope('conv_03') as scope:
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],padding="SAME", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3,rate=0.7, training=self.training)
                self.conv3_hist = tf.summary.histogram(self.name + "conv3", conv3)
                self.pool3_hist = tf.summary.histogram(self.name + "pool3", pool3)
                self.dropout3_hist = tf.summary.histogram(self.name + "dropout3", dropout3)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])

            with tf.name_scope('layer_01') as scope:
                dense4 = tf.layers.dense(inputs=flat,units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4,rate=0.5, training=self.training)
                self.dense4_hist = tf.summary.histogram(self.name + "dense4", dense4)
                self.dropout4_hist = tf.summary.histogram(self.name + "dropout4", dropout4)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            with tf.name_scope('layer_02') as scope:
                self.logits = tf.layers.dense(inputs=dropout4, units=10)
                self.logits_hist = tf.summary.histogram(self.name + "logits", self.logits)

        # define cost/loss & optimizer
        with tf.name_scope('cost') as scope:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.cost_sum = tf.summary.scalar(self.name + "cost", self.cost)

        with tf.name_scope('optimizer') as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        with tf.name_scope('accuracy') as scope:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracy_sum = tf.summary.scalar(self.name + "cost", self.accuracy)

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,feed_dict={self.X: x_test,self.Y: y_test, self.training: training})

    def train(self, merged_summary, x_data, y_data, training=True):
        return self.sess.run([merged_summary, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()

models = []
num_models = 1
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/9.6.mnist_cnn_ensemble")
writer.add_graph(sess.graph)  # Show the graph

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            summary, c, _ = m.train(merged_summary, batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
            writer.add_summary(summary, global_step=epoch * total_batch + i)

    test_size = len(mnist.test.labels)
    predictions = np.zeros(test_size * 10).reshape(test_size, 10)
    for m_idx, m in enumerate(models):
#        print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
        p = m.predict(mnist.test.images)
        predictions += p
    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list, 'Ensemble accuracy:', sess.run(ensemble_accuracy))

print('Learning Finished!')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

'''
0 Accuracy: 0.9933
1 Accuracy: 0.9946
2 Accuracy: 0.9934
3 Accuracy: 0.9935
4 Accuracy: 0.9935
5 Accuracy: 0.9949
6 Accuracy: 0.9941

Ensemble accuracy: 0.9952
'''


'''

'''