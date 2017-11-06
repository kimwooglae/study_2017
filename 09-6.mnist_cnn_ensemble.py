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
num_models = 10
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
C:\Users\user\AppData\Local\Programs\Python\Python35\python.exe C:/Users/user/PycharmProjects/study_2017/09-6.mnist_cnn_ensemble.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2017-10-30 13:58:05.340815: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-30 13:58:05.341046: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-30 13:58:05.594670: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1050 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.392
pciBusID 0000:01:00.0
Total memory: 4.00GiB
Free memory: 3.31GiB
2017-10-30 13:58:05.594903: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:976] DMA: 0 
2017-10-30 13:58:05.595038: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:986] 0:   Y 
2017-10-30 13:58:05.595159: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0)
Learning Started!
2017-10-30 13:58:51.937590: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.59GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:58:51.937890: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.34GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:58:52.223878: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.10GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:58:52.224145: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.37GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:58:52.224389: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.90GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:58:52.498224: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.75GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
Epoch: 0001 cost = [ 0.80431394] Ensemble accuracy: 0.9596
Epoch: 0002 cost = [ 0.28509821] Ensemble accuracy: 0.9761
Epoch: 0003 cost = [ 0.23210464] Ensemble accuracy: 0.9802
Epoch: 0004 cost = [ 0.20148074] Ensemble accuracy: 0.9819
Epoch: 0005 cost = [ 0.18430119] Ensemble accuracy: 0.984
Epoch: 0006 cost = [ 0.17207096] Ensemble accuracy: 0.9852
Epoch: 0007 cost = [ 0.16092415] Ensemble accuracy: 0.9862
Epoch: 0008 cost = [ 0.15403918] Ensemble accuracy: 0.9879
Epoch: 0009 cost = [ 0.14743359] Ensemble accuracy: 0.9888
Epoch: 0010 cost = [ 0.14735224] Ensemble accuracy: 0.9886
Epoch: 0011 cost = [ 0.14583699] Ensemble accuracy: 0.9896
Epoch: 0012 cost = [ 0.14240756] Ensemble accuracy: 0.9885
Epoch: 0013 cost = [ 0.13968648] Ensemble accuracy: 0.9901
Epoch: 0014 cost = [ 0.13711155] Ensemble accuracy: 0.9911
Epoch: 0015 cost = [ 0.1342607] Ensemble accuracy: 0.9915
Epoch: 0016 cost = [ 0.13178633] Ensemble accuracy: 0.991
Epoch: 0017 cost = [ 0.13141548] Ensemble accuracy: 0.991
Epoch: 0018 cost = [ 0.13468125] Ensemble accuracy: 0.9916
Epoch: 0019 cost = [ 0.12848745] Ensemble accuracy: 0.9919
Epoch: 0020 cost = [ 0.1277154] Ensemble accuracy: 0.9913
Learning Finished!
0 Accuracy: 0.9913
Ensemble accuracy: 0.9913

Process finished with exit code 0
'''