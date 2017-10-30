import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                  [[4],[5],[6]],
                  [[7],[8],[9]]]], dtype=np.float32)
'''
print("image.shape:", image.shape)  # (1, 3, 3, 1)

plt.imshow(image.reshape(3,3), cmap='Greys', interpolation='nearest')
plt.show()

weight = tf.constant([[[[1.,0.5]],[[1.,0.5]]],
                      [[[1.,0.5]],[[1.,0.5]]]])
print("weight.shape:", weight.shape)    # [Dimension(2), Dimension(2), Dimension(1), Dimension(2)]

conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID')
conv2d_img = conv2d.eval()
print("conv2d_img.shape:", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img,0,3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1)
    plt.imshow(one_img.reshape(2,2), cmap='gray', interpolation='nearest')

plt.show()


conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape:", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img,0,3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1)
    plt.imshow(one_img.reshape(3,3), cmap='gray', interpolation='nearest')

plt.show()

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])

print("weight.shape:", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()

print("conv2d_img.shape:", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap="gray", interpolation='nearest')
plt.show()

image = np.array([[[[4],[3]],
                  [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

print(pool.shape)
print(pool.eval())
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.show()

sess = tf.InteractiveSession()
img = img.reshape(-1,28,28,1)
W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')

print(conv2d)
'''
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1)
    plt.imshow(one_img.reshape(14,14), cmap='gray', interpolation='nearest')

plt.show()
'''
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(pool)

sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img,0,3)

for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1)
    plt.imshow(one_img.reshape(7,7), cmap='gray', interpolation='nearest')

plt.show()