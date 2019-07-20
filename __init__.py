# building convolutional neural network from scratch using only numpy
import layer
import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
from mlxtend.preprocessing import shuffle_arrays_unison
import pickle
# useful for debugging
np.random.seed(1)

# load out filters we saved on pickle
f1 = open("filter1.pickle", "rb")
f2 = open("filter2.pickle", "rb")
s0 = open("syn0.pickle", "rb")
s1 = open("syn1.pickle", "rb")
b0 = open("bias0.pickle", "rb")
b1 = open("bias1.pickle", "rb")

# load hyper parameters
filter1 = pickle.load(f1)
filter2 = pickle.load(f2)

syn0 = pickle.load(s0)
syn1 = pickle.load(s1)

bias0 = pickle.load(b0)
bias1 = pickle.load(b1)

# load MNIST dataset
mnist = fetch_mldata("MNIST original")

# learning rates for weights and bias
rate_syn = .0001
rate_bias = .0001

# data set
xarr = mnist.data
yarr = mnist.target

# shuffle arrays
xarr, yarr = shuffle_arrays_unison(arrays=[xarr, yarr], random_seed=4)
# only use first 1000 samples for training
xarr = xarr[:1000]
yarr = yarr[:1000]

# graphing variables
xbar = []
ybar = []
correct = 0
loss = 0

# training loop
for epoch in range(10):
    for image in range(1000):
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
        l0 = layer.dropout(l0, .5)
        z = layer.forward_connected(l0, syn0, bias0)
        l1 = layer.RELU(z)
        l1 = layer.dropout(l1, .5)
        l2 = layer.forward_connected(l1, syn1, bias1)
        l2 = layer.softmax(l2)
        # define target matrix
        target = np.zeros([10,1])
        target[int(yarr[image])][0] = 1
        # calculate cost
        loss += np.abs(layer.cost(l2, target))
#         print(str(loss) + " " + str(int(yarr[image])) + " " + str(np.argmax(l2)) +  " " + str(image))
        print(f'{loss} {int(yarr[image]} {np.argmax(l2)} {image}')
        
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
        # backprop convolutions from fully connected layer
        error_conv = np.dot(syn0.T, error_hidden)
        error_conv = layer.error_to_conv(error_conv)
        # compute change in filters
        error_max1 = layer.delta_pool(error_conv, relu1) * layer.RELU(conv1, True)
        delta_filter2 = layer.delta_filters(error_max1, max0, 5, 20)
        error_conv0 = layer.error_conv_layer(error_max1, filter2, 12)
        error_max0 = layer.delta_pool(error_conv0, relu0) * layer.RELU(conv0, True)
        delta_filter0 = layer.delta_filters(error_max0, input, 5, 10)
        
        # adjust filters
        filter1 -= .0001 * delta_filter0
        filter2 -= .0001 * delta_filter2
        # reset loss value
        loss = 0
    # adds average loss to graph
    ybar.append(loss / 1000)
    xbar.append(epoch)

# test network wih 150 samples randomly assigned
for i in range(150):
    # only forward propogate
    index = np.random.randint(0, 70000)
    input = layer.convert_to_2d_image(xarr[index])
    conv0 = layer.conv2d(input, filter1)
    relu0 = layer.RELU(conv0)
    max0 = layer.maxpool(relu0)
    conv1 = layer.conv2d(max0, filter2)
    relu1 = layer.RELU(conv1)
    max1 = layer.maxpool(relu1)
    l0 = layer.flatten(max1)
    z = layer.forward_connected(l0, syn0, bias0)
    l1 = layer.RELU(z)
    l2 = layer.forward_connected(l1, syn1, bias1)
    l2 = layer.softmax(l2)
    if np.argmax(l2) == yarr[index]:
        correct += 1

# prints number of correct out of testing samples
print(correct)
   
# save parameters after training
with open("filter1.pickle", "wb") as f1:
    pickle.dump(filter1, f1)

with open("filter2.pickle", "wb") as f2:
    pickle.dump(filter2, f2)

with open("syn0.pickle", "wb") as s0:
    pickle.dump(syn0, s0)

with open("syn1.pickle", "wb") as s1:
    pickle.dump(syn1, s1)

with open("bias0.pickle", "wb") as b0:
    pickle.dump(bias0, b0)

with open("bias1.pickle", "wb") as b1:
    pickle.dump(bias1, b1)

# show graph of progression of network
plt.plot(xbar, ybar, '-')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.suptitle("Cross Entropy loss time graph ConvNet")
plt.show()
    
