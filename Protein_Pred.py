import sys
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
import time
#import theano
import keras
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import tensorflow as tf
from keras.initializers import Initializer

#os.environ["cuda_visible_devices"] = "0,1,2"
#tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))



#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction=0.8

#sess = tf.Session(config=config)
#keras.backend.set_session(sess)

 
seed = 7
np.random.seed(seed)

#this is for force to CPU Only


from keras.utils import to_categorical

#import tensorflow as tf 

#num_cores = 1

#num_CPU = 1


#config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,inter_op_parallelism_threads=num_cores,allow_soft_placement=True,device_count = {'CPU': num_CPU})
#session = tf.Session(config=config)
#K.set_session(session)


#read the data set
print ("Read data {}".format (sys.argv[1]))
dataset = pd.read_csv(sys.argv[1])

#print(dataset)

#Spliting data into training and testing
X = dataset.iloc[:,1:261].values
#print(X)
Y = dataset.iloc[:,261].values
#print(Y)

#Label the class

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(Y)
#print(y1)
Y = pd.get_dummies(y1).values
#print(Y)


from sklearn.model_selection import train_test_split

#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=60)



from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD,Adam


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def baseline_model():

	model = Sequential()
	model.add(Dense(520,input_shape=(260,),activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(260,activation='relu',kernel_initializer='he_normal'))
	#model.add(BatchNormalization())
	model.add(Dense(130,activation='relu'))
	#model.add(BatchNormalization())
	model.add(Dense(65,activation='relu'))
	#model.add(BatchNormalization())
	model.add(Dense(32,activation='relu'))
	#model.add(BatchNormalization())
	model.add(Dense(16,activation='relu'))
	#model.add(BatchNormalization())
	#model.add(Dense(8,activation='relu'))
	#model.add(BatchNormalization())
	#model.add(Dense(4,activation='relu'))
	model.add(Dense(3,activation='softmax'))
	model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
	#model.summary()
	return model

#Fitting the model and predicting

start = time.time()
batch_size = int (sys.argv[3])
print ("Using batchsize = {}".format (batch_size))
estimator = KerasClassifier(build_fn=baseline_model, epochs=100,batch_size=batch_size)
#model.fit(X_train,Y_train,validation_split=0.3,epochs=500)

	
kfold  = KFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)


 
print("Average Acc: %.2f%% and StDev: (%.2f%%)" % (results.mean()*100, results.std()*100))

	
end = time.time()

#print(history.losses)


#Visualize the result

#from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(y_test_class,y_pred_class))
#print(confusion_matrix(y_test_class,y_pred_class))
print("elapse time is", end-start)

with open(sys.argv[2], 'w') as out:
	out.write ("Batch size : {}".format (batch_size))
	out.write("Average Acc: {:.2f} and StDev: {:.2f}\n".format (results.mean()*100, results.std()*100))
	out.write("elapse time is {:.2F}".format (end-start))

