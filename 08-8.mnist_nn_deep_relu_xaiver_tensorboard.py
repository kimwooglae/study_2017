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

with tf.name_scope('layer_01') as scope:
    W1 = tf.get_variable("weight1", shape=[784, 512],initializer=tf.contrib.layers.xavier_initializer())
#    W1 = tf.Variable(tf.random_normal([512,512]), name='weight1')
    b1 = tf.Variable(tf.random_normal([512]), name='bias1')
    l1 = tf.nn.relu(tf.matmul(X,W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    l1_hist = tf.summary.histogram("layer1", l1)


with tf.name_scope('layer_02') as scope:
    W2 = tf.get_variable("weight2", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
#    W2 = tf.Variable(tf.random_normal([512,512]), name='weight2')
    b2 = tf.Variable(tf.random_normal([512]), name='bias2')
    l2 = tf.nn.relu(tf.matmul(l1,W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    l2_hist = tf.summary.histogram("layer2", l2)

with tf.name_scope('layer_03') as scope:
    W3 = tf.get_variable("weight3", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
#    W3 = tf.Variable(tf.random_normal([512,512]))
    b3 = tf.Variable(tf.random_normal([512]))
    l3 = tf.nn.relu(tf.matmul(l2,W3) + b3)

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
    writer = tf.summary.FileWriter("./logs/8.8.mnist_nn_deep_relu_xaiver")
    writer.add_graph(sess.graph)  # Show the graph

    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, c, _ = sess.run([merged_summary, cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
            writer.add_summary(summary, global_step=epoch * total_batch + i)

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    # don't know why this makes Travis Build error.
    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()

'''
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/wlkim/Documents/workspace_tensorflow/study_2017/08-8.mnist_nn_deep_relu_xaiver_tensorboard.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2017-10-23 00:16:58.924999: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-23 00:16:58.925014: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-23 00:16:58.925018: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-23 00:16:58.925024: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
('Epoch:', '0001', 'cost =', '0.303468550')
('Epoch:', '0002', 'cost =', '0.100050543')
('Epoch:', '0003', 'cost =', '0.064405477')
('Epoch:', '0004', 'cost =', '0.046591868')
('Epoch:', '0005', 'cost =', '0.034507858')
('Epoch:', '0006', 'cost =', '0.030205952')
('Epoch:', '0007', 'cost =', '0.025886242')
('Epoch:', '0008', 'cost =', '0.020386845')
('Epoch:', '0009', 'cost =', '0.020509947')
('Epoch:', '0010', 'cost =', '0.017787809')
('Epoch:', '0011', 'cost =', '0.012461174')
('Epoch:', '0012', 'cost =', '0.018810769')
('Epoch:', '0013', 'cost =', '0.015535161')
('Epoch:', '0014', 'cost =', '0.011285567')
('Epoch:', '0015', 'cost =', '0.015233200')
Learning finished
('Accuracy: ', 0.9763)
('Label: ', array([9]))
('Prediction: ', array([9]))

Process finished with exit code 0
'''