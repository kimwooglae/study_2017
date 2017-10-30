# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

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
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.name_scope('conv_01') as scope:
                # L1 ImgIn shape=(?, 28, 28, 1)
                W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
#                W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
#                W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
                #    Conv     -> (?, 28, 28, 32)
                #    Pool     -> (?, 14, 14, 32)
                L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                self.conv1_hist = tf.summary.histogram("conv1", L1)

                '''
                Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
                Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
                Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
                Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
                '''

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope('conv_02') as scope:

                # L2 ImgIn shape=(?, 14, 14, 32)
                W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
#               W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
#               W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
                #    Conv      ->(?, 14, 14, 64)
                #    Pool      ->(?, 7, 7, 64)
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                self.conv2_hist = tf.summary.histogram("conv2", L2)
                '''
                Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
                Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
                Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
                Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
                '''

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.name_scope('conv_03') as scope:
                # L3 ImgIn shape=(?, 7, 7, 64)


                W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
#            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128]))
#            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
                #    Conv      ->(?, 7, 7, 128)
                #    Pool      ->(?, 4, 4, 128)
                #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
                L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.nn.relu(L3)
                L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                                    1, 2, 2, 1], padding='SAME')
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
                self.conv3_hist = tf.summary.histogram("conv3", L3)

                '''
                Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
                Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
                Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
                Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
                Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
                '''

            L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

            # L4 FC 4x4x128 inputs -> 625 outputs
            with tf.name_scope('layer_01') as scope:
                W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
                b4 = tf.Variable(tf.random_normal([625]))
                L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
                L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
                self.dropout4_hist = tf.summary.histogram("layer1", L4)

                '''
                Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
                Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
                '''

            # L5 Final FC 625 inputs -> 10 outputs
            with tf.name_scope('layer_02') as scope:
                W5 = tf.get_variable("W5", shape=[625, 10],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b5 = tf.Variable(tf.random_normal([10]))
                self.logits = tf.matmul(L4, W5) + b5
                self.logits_hist = tf.summary.histogram("logits", self.logits)
                '''
                Tensor("add_1:0", shape=(?, 10), dtype=float32)
                '''

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

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, merged_summary, x_data, y_data, keep_prop=0.7):
        return self.sess.run([merged_summary, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/9.4.mnist_cnn_class")
writer.add_graph(sess.graph)  # Show the graph

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, c, _ = m1.train(merged_summary, batch_xs, batch_ys)
        avg_cost += c / total_batch
        writer.add_summary(summary, global_step=epoch * total_batch + i)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

print('Learning Finished!')

# Test model and check accuracy
print('Final Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

'''
('Epoch:', '0001', 'cost =', '0.370451598')
('Epoch:', '0002', 'cost =', '0.100769491')
('Epoch:', '0003', 'cost =', '0.074093524')
('Epoch:', '0004', 'cost =', '0.061182146')
('Epoch:', '0005', 'cost =', '0.055214301')
('Epoch:', '0006', 'cost =', '0.047151379')
('Epoch:', '0007', 'cost =', '0.042627152')
('Epoch:', '0008', 'cost =', '0.040487051')
('Epoch:', '0009', 'cost =', '0.037241022')
('Epoch:', '0010', 'cost =', '0.036167392')
('Epoch:', '0011', 'cost =', '0.031483392')
('Epoch:', '0012', 'cost =', '0.030247647')
('Epoch:', '0013', 'cost =', '0.029072687')
('Epoch:', '0014', 'cost =', '0.027248394')
('Epoch:', '0015', 'cost =', '0.026338844')
Learning Finished!
('Accuracy:', 0.99379998)
'''


'''  epoch = 1
Learning Started!
('Epoch:', '0001', 'cost =', '0.370451598')
Learning Finished!
('Accuracy:', 0.97729999)
'''

''' epoch = 1, stddev delete
Learning Started!
('Epoch:', '0001', 'cost =', '47.637706182')
Learning Finished!
('Accuracy:', 0.15369999)
'''


''' epoch = 1, xaiver initializer
Learning Started!
('Epoch:', '0001', 'cost =', '0.301685038')
Learning Finished!
('Accuracy:', 0.98290002)
'''


'''
('Epoch:', '0001', 'cost =', '0.289042879', 'Accuracy:', 0.98500001)
('Epoch:', '0002', 'cost =', '0.082587733', 'Accuracy:', 0.98909998)
('Epoch:', '0003', 'cost =', '0.062591394', 'Accuracy:', 0.9892)
('Epoch:', '0004', 'cost =', '0.052383320', 'Accuracy:', 0.99059999)
('Epoch:', '0005', 'cost =', '0.044315733', 'Accuracy:', 0.99119997)
('Epoch:', '0006', 'cost =', '0.039941560', 'Accuracy:', 0.99269998)
('Epoch:', '0007', 'cost =', '0.037952311', 'Accuracy:', 0.99169999)
('Epoch:', '0008', 'cost =', '0.035166284', 'Accuracy:', 0.9932)
('Epoch:', '0009', 'cost =', '0.033917050', 'Accuracy:', 0.99220002)
('Epoch:', '0010', 'cost =', '0.030036120', 'Accuracy:', 0.9932)
('Epoch:', '0011', 'cost =', '0.029079973', 'Accuracy:', 0.99440002)
('Epoch:', '0012', 'cost =', '0.027750726', 'Accuracy:', 0.99440002)
('Epoch:', '0013', 'cost =', '0.024523864', 'Accuracy:', 0.99360001)
('Epoch:', '0014', 'cost =', '0.023976462', 'Accuracy:', 0.99309999)
('Epoch:', '0015', 'cost =', '0.023150473', 'Accuracy:', 0.99339998)
Learning Finished!
('Final Accuracy:', 0.99339998)
'''

'''
C:\Users\user\AppData\Local\Programs\Python\Python35\python.exe C:/Users/user/PycharmProjects/study_2017/09-4.mnist_deep_cnn_class.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2017-10-30 13:23:02.527879: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-30 13:23:02.528155: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-30 13:23:02.819392: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1050 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.392
pciBusID 0000:01:00.0
Total memory: 4.00GiB
Free memory: 3.31GiB
2017-10-30 13:23:02.819633: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:976] DMA: 0 
2017-10-30 13:23:02.819748: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:986] 0:   Y 
2017-10-30 13:23:02.820086: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0)
Learning Started!
2017-10-30 13:24:27.198147: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.59GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:24:27.198423: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.34GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:24:27.466028: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.10GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:24:27.514373: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.90GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2017-10-30 13:24:27.798108: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.75GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
Epoch: 0001 cost = 0.289006050 Accuracy: 0.9843
Epoch: 0002 cost = 0.083091439 Accuracy: 0.9894
Epoch: 0003 cost = 0.061275722 Accuracy: 0.989
Epoch: 0004 cost = 0.052544759 Accuracy: 0.9906
Epoch: 0005 cost = 0.046062796 Accuracy: 0.9917
Epoch: 0006 cost = 0.039569762 Accuracy: 0.9917
Epoch: 0007 cost = 0.036703118 Accuracy: 0.9911
Epoch: 0008 cost = 0.036134717 Accuracy: 0.9933
Epoch: 0009 cost = 0.032488339 Accuracy: 0.9914
Epoch: 0010 cost = 0.030079693 Accuracy: 0.9935
Epoch: 0011 cost = 0.028116586 Accuracy: 0.9935
Epoch: 0012 cost = 0.027264952 Accuracy: 0.9936
Epoch: 0013 cost = 0.025739732 Accuracy: 0.9933
Epoch: 0014 cost = 0.025085885 Accuracy: 0.9934
Epoch: 0015 cost = 0.024836152 Accuracy: 0.9944
Learning Finished!
Final Accuracy: 0.9944

Process finished with exit code 0

'''