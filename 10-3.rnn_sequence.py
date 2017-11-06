import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

#sample=" if you want you"
sample = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}
print ('idx2char:', idx2char)
print ('char2idx:',char2idx)

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]
print('sample_idx:', sample_idx)
print('x_data:', x_data)
print('y_data:', y_data)


# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1


X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

print('X_one_hot:', X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        l, _ =  sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        result_str = [idx2char[c] for c in np.squeeze(result)]
#        print("result", result)
#        print("np.squeeze(result)", np.squeeze(result))
#        print(result_str)
        print(i, "loss:", l, "Prediction:", ''.join(result_str))