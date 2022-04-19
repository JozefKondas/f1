import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np # linear algebra
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats

import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

##############this is function for my dataset, you probably won't need it #####################

def get_frames(df, frame_size, hop_size):
    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0,len(df )- frame_size, hop_size):
        x = df['x'].values[i: i+frame_size]
        y = df['y'].values[i: i+frame_size]
        z = df['z'].values[i: i+frame_size]
       # label = stats.mode(df['label'][i: i+frame_size])[0][0]
        label = min(df['label'][i: i+frame_size])

        #label = df['label'].values[i: i+frame_size]

	#label = (df['label'][i: i+frame_size])[0][0]

        frames.append([x,y,z])
        labels.append(label)
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    return frames, labels

#####################################

def get_dataset_partitions_pd(df, train_split=0.7, val_split=0.15, test_split=0.15):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1)#, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds
##################################################
if __name__ == "__main__":
  #frame size
    Fs=20
    frame_size = Fs*4 #80
    hop_size = Fs*2 #40
    # Load and compile Keras model
    model = Sequential()
    model.add(Conv2D(256, (2,2), activation = 'relu', input_shape = (80, 3, 1)))
    model.add(Dropout(0.1))
    model.add(Conv2D(512, (2,2), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset

#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


    dataframe = pd.read_csv("f1.csv", header=None)
    dataset = dataframe.values

    final = pd.DataFrame(data=dataset, columns=['x','y','z','label'])

############################
 

    final = final.iloc[1: , :]
    #X = dataset[:, 0:3]   #inputs
    #X = dataset[:,0:4]
    #Y = dataset[:,3]      #class (outputs)
    #Y = int(Y)
##################################################
#    train_ds, val_ds, test_ds=get_dataset_partitions_pd(final)

 #   x_train = train_ds[['x','y','z']]
  #  y_train = train_ds['label'].astype('float')

 #   x_val = val_ds[['x','y','z']]
  #  y_val = val_ds['label'].astype('float')

   # x_test = test_ds[['x','y','z']]
   # y_test = test_ds['label'].astype('float')
####################################################

    x = final[['x','y','z']]

    y = final['label'].astype('float')

    #print(x)
    print("---------------------------")
    #print(y)


    #scaler = StandardScaler()
    
    #x = scaler.fit_transform(x)
    x=x.astype(np.float)
################################################3
    #x=x.to_numpy()
    #x = tf.convert_to_tensor(x, dtype=tf.double) 
    #print(x)
###########################################################
   # scaler = StandardScaler()
   # x_train = scaler.fit_transform(x_train)
   # scaled_x_train = pd.DataFrame(data=x_train, columns=['x','y','z'])
   # scaled_x_train['label'] = y_train.values
   # x_train,y_train = get_frames(scaled_x_train, frame_size, hop_size)
   # print(x_train)
   # print(y_train)
   # print("*****************************************")

   # scaler = StandardScaler()
   # x_test = scaler.fit_transform(x_test)
   # scaled_x_test = pd.DataFrame(data=x_test, columns=['x','y','z'])
   # scaled_x_test['label'] = y_test.values
   # x_test,y_test = get_frames(scaled_x_test, frame_size, hop_size)

   # print(x_test)

   # scaler = StandardScaler()
    #x_val = scaler.fit_transform(x_val)
    #scaled_x_val = pd.DataFrame(data=x_val, columns=['x','y','z'])
    #scaled_x_val['label'] = y_val.values
    #x_val,y_val = get_frames(scaled_x_val, frame_size, hop_size)

    #print(x_val)

###########################################################


    scaled_x = pd.DataFrame(data=x, columns=['x','y','z'])
    scaled_x['label'] = y.values

    #print(scaled_x)


    #x,y = get_frames(final, frame_size, hop_size)
    x,y = get_frames(scaled_x, frame_size, hop_size)

################################################################################################
   # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state = 0)

    x_train, y_train = x,y


    # Use the last 5k training examples as a validation set
    x_test, y_test = x_train[40000:50000], y_train[40000:50000]

    x_train = x_train.reshape(x_train[:, :, :, np.newaxis].shape)
    x_test = x_test.reshape(x_test[:, :, :, np.newaxis].shape)

    # The `evaluate` function will be called after every round

####################################################################################

#########################################################
    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client , need to add IP adress:PORT of your server##################
    fl.client.start_numpy_client("192.168.1.31:8080", client=CifarClient())
