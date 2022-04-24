import pandas as pd
data=pd.read_csv('advertising.csv')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[['Daily_Time_ Spent _on_Site','Age','Area_Income','Daily Internet Usage']],data['Clicked_on_Ad'])
import keras.losses
import keras.optimizers
import keras.metrics
import keras.activations
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=data[['Daily_Time_ Spent _on_Site','Age','Area_Income','Daily Internet Usage']].shape[1],units=3,activation=keras.activations.sigmoid))
model.add(Dense(input_dim=data[['Daily_Time_ Spent _on_Site','Age','Area_Income','Daily Internet Usage']].shape[1],units=3,activation=keras.activations.sigmoid))
model.add(Dense(units=1,activation=keras.activations.sigmoid))
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
model.fit(x_train,y_train,batch_size=10,epochs=10)
pred=model.predict([[69.57,48,51636.92,113.12]])
for i in pred:
    if i[0] > 0.5:
        print('yes clicked ')
    else:
        print('no not clicked')