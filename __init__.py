# building convolutional neural network from scratch using only numpy

import layer
import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
from mlxtend.preprocessing import shuffle_arrays_unison
import pickle
# useful for debugging
np.random.seed(1)

# initializing our 2 kernels / filters
# multiply by 2 because I wanted greater variance in random distribution
# 10 is the number of separate filters
# 1 is the depth
# 5x5 is the dimensions of the kernel
f1 = open("filter1.pickle", "rb")
f2 = open("filter2.pickle", "rb")
s0 = open("syn0.pickle", "rb")
s1 = open("syn1.pickle", "rb")
b0 = open("bias0.pickle", "rb")
b1 = open("bias1.pickle", "rb")

filter1 = pickle.load(f1)
# 20 filters, each 5x5
# deoth of 10
filter2 = pickle.load(f2)

# adjust shape of weights for fully connected so that 
# it passes the values to the next layer properly
syn0 = pickle.load(s0)
syn1 = pickle.load(s1)

bias0 = pickle.load(b0)
bias1 = pickle.load(b1)

# load MNIST dataset
mnist = fetch_mldata("MNIST original")

rate_syn = .0001
rate_bias = .0001

xarr = mnist.data
yarr = mnist.target

xarr, yarr = shuffle_arrays_unison(arrays=[xarr, yarr], random_seed=4)
xarr = xarr[500:1000]
yarr = yarr[500:1000]

xbar = []
ybar = []
correct = 0

for epoch in range(2):
    for image in range(500):
    # convolution operation with max pooling
        input = layer.convert_to_2d_image(xarr[image])
        conv0 = layer.conv2d(input, filter1)
        relu0 = layer.RELU(conv0)
        max0 = layer.maxpool(relu0)
        conv1 = layer.conv2d(max0, filter2)
        relu1 = layer.RELU(conv1)
        max1 = layer.maxpool(relu1)
        # fully connected
        l0 = layer.flatten(max1)
        z = layer.forward_connected(l0, syn0, bias0)
        l1 = layer.RELU(z)
        l2 = layer.forward_connected(l1, syn1, bias1)
        l2 = layer.softmax(l2)
        # define target matrix
        target = np.zeros([10,1])
        target[int(yarr[image])][0] = 1
        loss = layer.cost(l2, target)
        ybar.append(loss)
        xbar.append(image + epoch * 500)
        print(str(loss) + " " + str(int(yarr[image])) + " " + str(np.argmax(l2)) +  " " + str(image))
        if int(yarr[image]) == np.argmax(l2):
            correct += 1
        # backprop fully connected
        syn1_delta = np.outer((l2 - target), l1.T) 
        bias1_delta = (l2 - target)
        error_hidden = np.dot(syn1.T, (l2 - target)) * layer.RELU(z, True)
        syn0_delta = np.outer(error_hidden,l0.T)
        bias0_delta = error_hidden
        # update fully connected
        syn0 -= rate_syn * syn0_delta
        syn1 -= rate_syn * syn1_delta
        bias0 -= rate_bias * bias0_delta
        bias1 -= rate_bias * bias1_delta
        #backprop convolutions
        error_conv = np.dot(syn0.T, error_hidden)
        error_conv = layer.error_to_conv(error_conv)
        # not sure where to apply relu deriv
        error_max1 = layer.delta_pool(error_conv, relu1) * layer.RELU(conv1, True)
        delta_filter2 = layer.delta_filters(error_max1, max0, 5, 20)
        error_conv0 = layer.error_conv_layer(error_max1, filter2, 12)
        error_max0 = layer.delta_pool(error_conv0, relu0) * layer.RELU(conv0, True)
        delta_filter0 = layer.delta_filters(error_max0, input, 5, 10)
        
        filter1 -= .0001 * delta_filter0
        filter2 -= .0001 * delta_filter2
        
    print("Correct after epoch: " + str(correct))
    correct = 0
   


f1 = open("filter1.pickle", "wb")
pickle.dump(filter1, f1)
f1.close()

f2 = open("filter2.pickle", "wb")
pickle.dump(filter2, f2)
f2.close()

s0 = open("syn0.pickle", "wb")
pickle.dump(syn0, s0)
s0.close()

s1 = open("syn1.pickle", "wb")
pickle.dump(syn1, s1)
s1.close()

b0 = open("bias0.pickle", "wb")
pickle.dump(bias0, b0)
b0.close()

b1 = open("bias1.pickle", "wb")
pickle.dump(bias1, b1)
b1.close()

plt.plot(xbar, ybar, '-')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.suptitle("Cross Entropy loss time graph ConvNet")
plt.show()
    