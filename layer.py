import numpy as np

# method that performs the convolution operation
def conv2d(input, filter, stride = 1, padding = 0):
    # input is the volume you perform convolution operation on
    # filter is the kernel to use to convolution
    # default parameters side of 1, and padding of 0
    
    # compares depth if they are the same or not
    if input.shape[0] != filter.shape[1]:
        raise ValueError('depth don\'t match')
    # initialize important values for convolution operation
    depth = input.shape[0]
    numfilters = filter.shape[0]
    output_width = int((input.shape[1] + (2 * padding) - filter.shape[2]) / stride) + 1
    # setting output size
    output_layer = np.zeros([numfilters, output_width, output_width])
    # i don't plan to use padding, so there will be no case where you have to pad the input
    # but to that is very simple
    
    # convolution operation
    for f in range(numfilters):
        for d in range(depth):
            for dim1 in range(output_width):
                for dim2 in range(output_width):
                    for row in range(filter.shape[2]):
                        for col in range(filter.shape[2]):
                            output_layer[f][dim1][dim2] += input[d][row + dim1][col + dim2] * filter[f][d][row][col]
                    
    return output_layer
    
# finds max between four values and retusn value
def findMaxValue(num1, num2, num3, num4):
    return np.array([num1, num2, num3, num4]).max()

# max pooling method
def maxpool(input, filtersize = 2,stride = 2):
    # input is the volume to perform maxpooling
    # filter size is the width or heigh - this filter will always be a square matrix
    # stride is how much it skips, so it will perform pooling every two columns
    output_width = int(input.shape[1] / filtersize)
    depth = input.shape[0]
    output_layer = np.zeros([depth, output_width, output_width])
    
    r = 0
    c = 0
    for d in range(depth):
        for row in range(output_width):
            for col in range(output_width):
                output_layer[d][row][col] = findMaxValue(input[d][r][c],
                                                         input[d][r + 1][c],
                                                         input[d][r][c + 1],
                                                         input[d][r + 1][c + 1])
                c += stride
            c = 0
            r += stride
        r = 0
        c = 0
    
    
    return output_layer

# used to connect from 3D input to fully connected layers
def flatten(input):
    return input.reshape(input.shape[0] * input.shape[1] * input.shape[2], 1)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

# activation function
def RELU(matrix, deriv = False):
    if(deriv == True):
        return dReLU(matrix)
    else:
        return ReLU(matrix)

# forward propogate function
def forward_connected(matrix, weight, bias):
    return np.dot(weight, matrix) + bias

# outputs set of probabilistic values
def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()
    
# converts array to 2d arrays
def convert_to_2d_image(arr):
    height = int(np.sqrt(arr.shape[0]))
    count = 0
    image = np.array([np.zeros(height) for i in range(height)])
    for i in range(height):
        for x in range(height):
            image[i][x] = arr[count]
            count += 1
    # this is to make the image have a depth of 1
    return np.array([image])

# error that passes from the fully connected layers to the convolutional layers
def error_to_conv(vector):
    tensor = np.zeros((20,4,4))
    index = 0
    for i in range(20):
        for j in range(4):
            for k in range(4):
                tensor[i][j][k] = vector[index]
                index += 1
    return tensor

# cross entropy function
def cross_entropy(output, y_target):
    return - np.sum(np.log(output) * (y_target))

# calculates cross entropy loss
def cost(output, y_target):
    return np.mean(cross_entropy(output, y_target))

# returns index of max value
def indexMaxValue(num1, num2, num3, num4):
    return np.array([num1, num2, num3, num4]).argmax()

# this method passes the gradients from a pooled layer
def delta_pool(error, matrix, stride = 2):
    output = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[2]))
    for d in range(error.shape[0]):
        for row in range(error.shape[1]):
            for col in range(error.shape[2]):
                temp = indexMaxValue(matrix[d][row * 2][col * 2],
                                     matrix[d][row * 2 + 1][col * 2],
                                     matrix[d][row * 2][col * 2 + 1],
                                     matrix[d][row * 2 + 1][col * 2 + 1])
                if temp == 0:
                    output[d][row * 2][col * 2] = error[d][row][col]
                elif temp == 1:
                    output[d][row * 2 + 1][col * 2] = error[d][row][col]
                elif temp == 2:
                    output[d][row * 2][col * 2 + 1] = error[d][row][col]
                else:
                    output[d][row * 2 + 1][col * 2 + 1] = error[d][row][col]
                
    return output

# this method computes how much the filter should change by
def delta_filters(error, input_prev, weight_width, numfilters):
    delta_filter = np.zeros((numfilters, input_prev.shape[0], weight_width, weight_width))
    depth = input_prev.shape[0]
    
    for f in range(numfilters):
        for d in range(depth):
            for row in range(error.shape[1]):
                for col in range(error.shape[2]):
                    for r in range(weight_width):
                        for c in range(weight_width):
                            delta_filter[f][d][r][c] += error[f][row][col] * input_prev[d][r + row][c + col]
    
    return delta_filter

# this method passes the layer to the previous convolution layer
def error_conv_layer(error, weight, output_width):
    output = np.zeros((weight.shape[1], output_width, output_width))
    numfilters = weight.shape[0]
    depth = weight.shape[1]
    # depth is also numb filters
    for f in range(numfilters):
        for d in range(depth):
            for row in range(error.shape[1]):
                for col in range(error.shape[2]):
                    for r in range(weight.shape[2]):
                        for c in range(weight.shape[3]):
                            output[d][row + r][col + c] += error[f][row][col] * weight[f][d][r][c]
    
    return output
        
    
    
    
