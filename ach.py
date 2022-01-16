import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
import random
from sklearn.model_selection import train_test_split

labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def load_dataset(path):

    with open(path,'rb') as data:
        dataset = pickle.load(data)

    for key, value in dataset.items() :
    	print(key)

    x = np.array(dataset['feature'][0]).reshape(372038,28,28)
    x = x.astype('float32')
    x = x/255.0
    x = x[..., np.newaxis]
    y = np.array(dataset['label'][0])
    y = y.astype('int')
    print(x.shape)
    print(y.shape)

    return x,y

img,label = load_dataset("C:/Users/ASUS/Documents/Program/Python/AI/Computer Vision/Character Recognition/data.pickle")
x_train,x_test,y_train,y_test = train_test_split(img,label,test_size=0.6)
print(x_test.shape)

OD1 = tf.keras.models.load_model('ocr_test')
OD1.summary()
OD2 = tf.keras.models.load_model('ocr_test3')
OD2.summary()
OD3 = tf.keras.models.load_model('ocr_test4')
OD3.summary()

start_time = time.time()
right = 0
for i in range(10000):
    rand = random.randint(1,32512)
    feature_pred = [x_test[rand]]
    feature_pred = np.array(feature_pred)
    prediction = OD1.predict(feature_pred)
    top_k_values, top_k_indices = tf.nn.top_k(prediction)
    #print(top_k_indices," ",top_k_values)
    an_array = top_k_indices.numpy()
    #print(f"Actual Label : {labels[y_test[rand]]} | Predicted Label : {labels[an_array[0][0]]}")
    if labels[y_test[rand]] == labels[an_array[0][0]]:
        right += 1
    print(f"Predicting Label : {i} / 10000", end="\r")
plt.show()
print(right)
print(f"\nPrediction Time {(time.time() - start_time)} seconds | Acc : {right/10000}")

start_time = time.time()
right = 0
for i in range(10000):
    rand = random.randint(1,32512)
    feature_pred = [x_test[rand]]
    feature_pred = np.array(feature_pred)
    prediction = OD2.predict(feature_pred)
    top_k_values, top_k_indices = tf.nn.top_k(prediction)
    #print(top_k_indices," ",top_k_values)
    an_array = top_k_indices.numpy()
    #print(f"Actual Label : {labels[y_test[rand]]} | Predicted Label : {labels[an_array[0][0]]}")
    if labels[y_test[rand]] == labels[an_array[0][0]]:
        right += 1
    print(f"Predicting Label : {i} / 10000", end="\r")
plt.show()
print(right)
print(f"\nPrediction Time {(time.time() - start_time)} seconds | Acc : {right/10000}")

start_time = time.time()
right = 0
for i in range(10000):
    rand = random.randint(1,32512)
    feature_pred = [x_test[rand]]
    feature_pred = np.array(feature_pred)
    prediction = OD3.predict(feature_pred)
    top_k_values, top_k_indices = tf.nn.top_k(prediction)
    #print(top_k_indices," ",top_k_values)
    an_array = top_k_indices.numpy()
    #print(f"Actual Label : {labels[y_test[rand]]} | Predicted Label : {labels[an_array[0][0]]}")
    if labels[y_test[rand]] == labels[an_array[0][0]]:
        right += 1
    print(f"Predicting Label : {i} / 10000", end="\r")
plt.show()
print(right)
print(f"\nPrediction Time {(time.time() - start_time)} seconds | Acc : {right/10000}")