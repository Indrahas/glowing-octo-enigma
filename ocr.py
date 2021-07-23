# p = initialize_parameters()
# for b in boxes:
#     print(predict(b, p))


# In[129]:


pip install opencv-python


# In[240]:


import pandas as pd
from PIL import Image
headers = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen']
note = pd.read_csv("TRNind.csv",names = headers)
note1 = pd.read_csv("TSTind.csv",names = headers)
headers1 = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','ninteen','twenty','twentyone','22','23','24','25','26','27','28','29','30','31','32','33','34']
names = pd.read_csv("ALLnames.csv")
labels = pd.read_csv("ALLlabels.csv",names = ['label'])
print(len(names))


# In[241]:


import random
samples = list(names['names'])
y = list(labels['label'])
new_samples = []
for i in range(len(samples)):
    new_samples.append((y[i],samples[i]))
random.shuffle(new_samples)
index = (4*len(new_samples))//5
training = new_samples[:index]
testing = new_samples[index:]
print(len(new_samples),len(testing))
print(training[0][1])


# In[242]:


import numpy as np
x_training=[]
y_training = []
x_testing=[]
y_testing=[]
for i in training:
    img=Image.open(r"F:/Indra/New folder/English/Img/"+i[1]+".png").resize((64,64))
    img.load()
    data = np.asarray(img,dtype ='int32')
    if data.shape == (64,64,3):
        x_training.append(data)
        y_training.append(i[0])
for i in testing:
    img=Image.open(r"F:/Indra/New folder/English/Img/"+i[1]+".png").resize((64,64))
    img.load()
    data = np.asarray(img,dtype ='int32')
    if data.shape == (64,64,3):
        x_testing.append(data)
        y_testing.append(i[0])
x_training_arr = np.asarray(x_training).reshape(len(x_training),64,64,3)
x_testing_arr = np.asarray(x_testing).reshape(len(x_testing),64,64,3)
y_training_arr = np.asarray(y_training).reshape(1,len(y_training))
y_testing_arr = np.asarray(y_testing).reshape(1,len(y_testing))
print(x_training_arr.shape,x_testing_arr.shape)


# In[243]:


def load_dataset():

    train_set_x_orig = x_training_arr # your train set features
    train_set_y_orig = y_training_arr # your train set labels


    test_set_x_orig = x_testing_arr # your test set features
    test_set_y_orig = y_testing_arr # your test set labels
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


# In[244]:


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# In[245]:


#OCR
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32,shape = [None, n_H0, n_W0, n_C0],name="X")
    Y = tf.placeholder(tf.float32,shape = [None, n_y])
    return X, Y


# In[246]:


index = 30
plt.imshow(X_train_orig[index])
print (X_train_orig.shape)
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# In[247]:


X_train = X_train_orig/255.
X_test = X_test_orig/255.
print(len(Y_train_orig.reshape(-1)))
Y_train = convert_to_one_hot(Y_train_orig, 63).T
Y_test = convert_to_one_hot(Y_test_orig, 63).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


# In[292]:


def initialize_parameters(beta = 0.009):
    W1 = tf.get_variable("W1",[4,4,3,8],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b1= tf.get_variable("b1", [8], initializer=tf.contrib.layers.xavier_initializer())
    b2= tf.get_variable("b2",[16], initializer=tf.contrib.layers.xavier_initializer())
    parameters = {"W1": W1,"W2": W2,"b1":b1,"b2":b2}
    if beta != 0:
        regularizer = tf.contrib.layers.l2_regularizer(scale = beta)
    else:
        regularizer = None
    return parameters,regularizer


# In[293]:


def forward_propagation(X, parameters,regularizer = None,keep_prob = 0.5):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters["b1"]
    b2 = parameters['b2']
    Z1 = tf.nn.conv2d(X,W1,strides = [1,1,1,1], padding = "SAME")
    B1 = tf.nn.bias_add(Z1, b1)
    A1 = tf.nn.relu(B1)
    D1 = tf.nn.dropout(A1, keep_prob)
    P1 = tf.nn.max_pool(D1,ksize=[1,8,8,1],strides = [1,8,8,1],padding = 'SAME')
    Z2 = tf.nn.conv2d(P1,W2,strides = [1,1,1,1] ,padding="SAME")
    B2 = tf.nn.bias_add(Z2, b2)
    A2 = tf.nn.relu(B2)
    D2 = tf.nn.dropout(A2, keep_prob)
    P2 = tf.nn.max_pool(D2,ksize = [1,4,4,1],strides = [1,4,4,1],padding="SAME")
    F = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(F,63,activation_fn = None,weights_regularizer=regularizer)
    return Z3


# In[294]:


def compute_cost(Z3, Y,regularizer = None):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3 , labels =Y))
    if regularizer is not None:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer,reg_variables)
    else:
        reg_term = 0
    return cost+reg_term


# In[295]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# In[296]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.006,
          num_epochs = 110, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    # Create Placeholders of the correct shape
    X, Y = create_placeholders( n_H0, n_W0, n_C0,n_y)
    # Initialize parameters
    parameters,regularizer = initialize_parameters()
    # Forward propagation
    Z3 = forward_propagation(X,parameters,regularizer)
    # Cost function
    cost = compute_cost(Z3, Y,regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run(
                                                fetches=[optimizer, cost],
                                                feed_dict={X: minibatch_X,
                                                           Y: minibatch_Y}
                                                )
                minibatch_cost += temp_cost / num_minibatches
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy*100)
        print("Test Accuracy:", test_accuracy*100)
        saver = tf.train.Saver()
        tf.add_to_collection('predict_op', predict_op)
        saver.save(sess, './my-test-model')
        return train_accuracy, test_accuracy, parameters


# In[297]:


_, _, parameters = model(X_train, Y_train, X_test, Y_test)


# In[235]:


# path = "F:/Indra/New folder/English/Img/"+"GoodImg/Bmp/Sample001/img001-00004"+".png"
# x_pre=[]
# img=Image.open(r"F:/Indra/New folder/English/Img/"+"GoodImg/Bmp/Sample005/img005-00009"+".png").resize((64,64))
# img.load()
# data = np.asarray(img,dtype ='int32')
# if data.shape == (64,64,3):
#     x_pre.append(data)
# x_predict_arr = np.asarray(x_pre).reshape(1,64,64,3)
# x_predict = np.asarray(x_pre)
# print(data.shape)


# In[236]:

def predict(x_arr):
    labels = {1:"0",2:"1",3:"2",4:"3",5:"4",6:"5",7:"6",8:"7",9:"8",10:"9",
              11:"A",12:"B",13:"C",14:"D",15:"E",16:"F",17:"G",18:"H",19:"I",20:"J",
              21:"K",22:"L",23:"M",24:"N",25:"O",26:"P",27:"Q",28:"R",29:"S",30:"T",
              31:"U",32:"V",33:"W",34:"X",35:"Y",36:"Z",37:"a",38:"b",39:"c",40:"d",
              41:"e",42:"f",43:"g",44:"h",45:"i",46:"j",47:"k",48:"l",49:"m",50:"n",
              51:"o",52:"p",53:"q",54:"r",55:"s",56:"t",57:"u",58:"v",59:"w",60:"x",
              61:"y",62:"z",0:"None"
             }
    plt.imshow(data)
    tf.reset_default_graph()
    checkpoint_path="F:/Indra/New folder/Datasets/my-test-model" #Write your path for .meta file
    with tf.Session() as sess:

    ## Load the entire model previuosly saved in a checkpoint
        print("Load the model from path", checkpoint_path)
        the_Saver = tf.train.import_meta_graph(checkpoint_path+".meta")
        the_Saver.restore(sess, checkpoint_path)

        ## Identify the predictor of the Tensorflow graph
        predict_op = tf.get_collection('predict_op')[0]

        ## Identify the restored Tensorflow graph
        dataFlowGraph = tf.get_default_graph()

        ## Identify the input placeholder to feed the images into as defined in the model
        x = dataFlowGraph.get_tensor_by_name("X:0")

        ## Predict the image category
        prediction = sess.run(predict_op, feed_dict = {x: x_arr})
        val = int(np.squeeze(prediction))
        print("nThe predicted image class is:", labels[val])
        return labels[val]


# In[ ]:
