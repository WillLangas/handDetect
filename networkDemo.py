import numpy
import scipy.special

import matplotlib.pyplot
%matplotlib inline

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
    
        
        #weights inside the array are w_i_j. Link is from node i to j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
        
        self.lr = learningrate
        
        #creates our activation function through scipy
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
   
    def train(self, inputs_list, targets_list):
        
        #alters the lists of inputs and targets into arrays 
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T
        
        #same as in the query function. Passes the signal through the hidden layer and moderating weights 
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #applies weights and passes signal through the output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        #error is (target - actual) for output error
        output_errors = targets - final_outputs
        
        #hidden layer error is the output_errors, split by the weights, combined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #update link weights between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #update link weights between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)), numpy.transpose(inputs))                                  
        pass
    
    #takes an input and provides an output
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.15

# create instance of the neural network class
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#import the MNIST training data file. Convert into a list
training_data_file = open("Desktop/neuralNetwork/trainingData/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network
for record in training_data_list:
    all_values = record.split(',') #split the records by the commas
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #convert the values into floats and scale down + shift
    targets = numpy.zeros(output_nodes) + 0.01 #create the target output values list
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass
 
test_data_file = open("Desktop/neuralNetwork/trainingData/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = [] #will show how well the network performs
for record in test_data_list: #sets the range of the dataset
    all_values = record.split(',') #split the list by the commas
    correct_label = int(all_values[0]) #finds the target answer
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #scales the rest of the inputs
    outputs = n.query(inputs) #performs the network calculcations on the inputs
    label = numpy.argmax(outputs) #finds the maximum of the output nodes, the network' "guess"
    if(label == correct_label): #adds a point if correct, adds nothing if incorrect
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
 
 #Demo Code:
all_values = record.split(',') #split the list by the commas
correct_label = int(all_values[0]) #finds the target answer
inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #scales the rest of the inputs
outputs = n.query(inputs) #performs the network calculcations on the inputs
label = numpy.argmax(outputs) #finds the maximum of the output nodes, the network' "guess"
print("Target: ", correct_label)
print("Guess: ", label)
if(label == correct_label):
    print("Network was correct")
else:
    print("Network was incorrect")
    pass

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap = 'Reds', interpolation = 'None')
print(outputs)

scorecard_array = numpy.asfarray(scorecard)
performance = (scorecard_array.sum()/scorecard_array.size) * 100
print('performance: ', performance, "%") #finds the amount correct as a percentage
