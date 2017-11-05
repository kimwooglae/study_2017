import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
sequence_length = 10  # any arbitrary number

char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}
print ('char_set:', char_set)
print ('char_dic:', char_dic)

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1:i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

# sentence_idx = [char_dic[c] for c in sentence]
# x_data = [sentence_idx[:-1]]
# y_data = [sentence_idx[1:]]
# print('sample_idx:', sentence_idx)
# print('x_data:', x_data)
# print('y_data:', y_data)


# hyper parameters

data_dim = len(char_set)  # RNN input size (one hot size)
hidden_size = len(char_set)  # RNN output size
num_classes = len(char_set)  # final output size (RNN or softmax, etc.)

batch_size = len(dataX)
learning_rate = 0.1

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

print('X_one_hot:', X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell]*5, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])

outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])


weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        l, _, results = sess.run([loss, train, outputs], feed_dict={X: dataX, Y: dataY})
#        print("result", result)
        print(i, "loss:", l)
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), l)
