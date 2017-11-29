import tensorflow as tf
import numpy as np

tf.set_random_seed(777)


def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


seq_length = 21
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 1000



xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
xy = MinMaxScaler(xy)

x = xy
y = xy[:, [-1]]

#print('x ', x)
#print('y ', y)

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
#    print(_x, '->', _y)
    dataX.append(_x)
    dataY.append(_y)


train_size = int(len(dataY)* 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
print(X)
Y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)

loss = tf.reduce_mean(tf.square(Y_pred - Y))

train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        l, _ = sess.run([loss, train], feed_dict={X: trainX, Y:trainY})
        print(i, l)
    testPredict = sess.run(Y_pred, feed_dict={X:testX})

    import matplotlib.pyplot as plt
    plt.plot(testY)
    plt.plot(testPredict)
    plt.show()