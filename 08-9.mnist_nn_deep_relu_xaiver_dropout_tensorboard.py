import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10

X = tf.placeholder(tf.float32, [None,784], name='x-input')
Y = tf.placeholder(tf.float32, [None,nb_classes], name='y-input')
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer_01') as scope:
    W1 = tf.get_variable("weight1", shape=[784, 512],initializer=tf.contrib.layers.xavier_initializer())
#    W1 = tf.Variable(tf.random_normal([512,512]), name='weight1')
    b1 = tf.Variable(tf.random_normal([512]), name='bias1')
    l1 = tf.nn.relu(tf.matmul(X,W1) + b1)
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    l1_hist = tf.summary.histogram("layer1", l1)


with tf.name_scope('layer_02') as scope:
    W2 = tf.get_variable("weight2", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
#    W2 = tf.Variable(tf.random_normal([512,512]), name='weight2')
    b2 = tf.Variable(tf.random_normal([512]), name='bias2')
    l2 = tf.nn.relu(tf.matmul(l1,W2) + b2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    l2_hist = tf.summary.histogram("layer2", l2)

with tf.name_scope('layer_03') as scope:
    W3 = tf.get_variable("weight3", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
#    W3 = tf.Variable(tf.random_normal([512,512]))
    b3 = tf.Variable(tf.random_normal([512]))
    l3 = tf.nn.relu(tf.matmul(l2,W3) + b3)
    l3 = tf.nn.dropout(l3, keep_prob=keep_prob)

    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    l3_hist = tf.summary.histogram("layer3", l3)

with tf.name_scope('layer_04') as scope:
    W4 = tf.get_variable("weight4", shape=[512, nb_classes],initializer=tf.contrib.layers.xavier_initializer())
#    W4 = tf.Variable(tf.random_normal([512,nb_classes]))
    b4 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(l3, W4) + b4

    w4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("biases4", b4)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)
    
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope('optimizer') as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_sum = tf.summary.scalar('accuracy', accuracy)

# parameters
#training_epochs = 5
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/8.9.mnist_nn_deep_relu_xaiver_dropout")
    writer.add_graph(sess.graph)  # Show the graph

    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, c, _ = sess.run([merged_summary, cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys, keep_prob : 0.7})
            avg_cost += c / total_batch
            writer.add_summary(summary, global_step=epoch * total_batch + i)

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels, keep_prob : 1.0}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob:1.0}))

    # don't know why this makes Travis Build error.
    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()

'''
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/wlkim/Documents/workspace_tensorflow/study_2017/08-9.mnist_nn_deep_relu_xaiver_dropout_tensorboard.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2017-10-23 00:26:18.123082: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-23 00:26:18.123103: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-23 00:26:18.123111: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-23 00:26:18.123120: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
('Epoch:', '0001', 'cost =', '0.618761646')
('Epoch:', '0002', 'cost =', '0.237850197')
('Epoch:', '0003', 'cost =', '0.183601648')
('Epoch:', '0004', 'cost =', '0.153045738')
('Epoch:', '0005', 'cost =', '0.136643496')
('Epoch:', '0006', 'cost =', '0.126910209')
('Epoch:', '0007', 'cost =', '0.113631939')
('Epoch:', '0008', 'cost =', '0.110915590')
('Epoch:', '0009', 'cost =', '0.101804307')
('Epoch:', '0010', 'cost =', '0.099059579')
('Epoch:', '0011', 'cost =', '0.093637820')
('Epoch:', '0012', 'cost =', '0.085889680')
('Epoch:', '0013', 'cost =', '0.086210292')
('Epoch:', '0014', 'cost =', '0.081376265')
('Epoch:', '0015', 'cost =', '0.078109609')
Learning finished
('Accuracy: ', 0.98159999)
('Label: ', array([1]))
('Prediction: ', array([1]))

Process finished with exit code 0
'''