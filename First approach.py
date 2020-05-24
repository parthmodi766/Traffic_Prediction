# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:21:45 2020

@author: Asus
"""

import h5py
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train_df= pd.read_csv('Data.csv')
print(train_df)
x = train_df[['stationid','day_of_month','month','year','hour','day_of_week','flow_0','flow_5','flow_10','flow_15','flow_20','flow_25','flow_30','flow_35','flow_40','flow_45','flow_50','flow_55']]
y = train_df[['flow_0','flow_5','flow_10','flow_15','flow_20','flow_25','flow_30','flow_35','flow_40','flow_45','flow_50','flow_55']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
model = Sequential()
model.add(Dense(85,activation= 'relu' , input_shape=(18,)))
model.add(Dense(425,activation= 'relu'))
model.add(Dense(425,activation= 'relu'))
model.add(Dense(85,activation= 'relu'))
model.add(Dense(12,))
model.compile(optimizer='adam', loss= 'mean_absolute_error', metrics=[ 'accuracy'])
early_stopping_monitor = EarlyStopping(patience=3)
history = model.fit(x_train,y_train, validation_split=0.1, epochs=2)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)

model.save('traffic_prediction.h5')
new_model = tf.keras.models.load_model('traffic_prediction.h5')
predictions = new_model.predict([x_test])
print(predictions.shape)
print(predictions)
y_true=y_test/1000
y_pred=predictions/1000


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
