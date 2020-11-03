#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## this file originated from the tutorial from sentdex on youtube
## with personal changes to the layer structure as well as how data
## is read in so i can use current data from yahoo finance using the
## script getdata

import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import glob


#predicts what price is in 10 candles from the last 100
SEQ_LEN = 100 
FUTURE_PERIOD_PREDICT = 10
RATIO_TO_PREDICT = 'TSLA'
EPOCHS = 100
BATCH_SIZE = 50
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{-FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

#classifies prediction between higher or lower than current candle
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

#normalizes the data
def process_df_pcts(df):
    for col in df.columns:
        if col != "target" and col != "future":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    return df


def preprocess_df(df):
    df = df.drop('future', 1)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X=[]
    y=[]
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y

main_df = pd.DataFrame()

symbols = ["AAPL"]

for symbol in symbols:
    path = f"MarketData/{symbol}" # use your path
    all_files = glob.glob(path + "/*.csv")
    #dataset = f"MarketData/{symbols}/.csv"
    df = pd.concat((pd.read_csv(f, header=0) for f in all_files))
    #df = pd.read_csv(dataset,names=["time", "open", "high", "low", "close", "volume"])
    
    #df.rename(columns={"close": "close", "volume":"BTC-USD_volume"}, inplace=True)
   
    df.set_index("Datetime", inplace=True)
    df = df[[f"Close", f"Volume", "Open", "High", "Low"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df["Close"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df["Close"], main_df["future"]))
print(main_df.head(10))
main_df = process_df_pcts(main_df)


times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)] #out of sequence data
main_df = main_df[(main_df.index < last_5pct)] #in sequence data

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y  = preprocess_df(validation_main_df)


print(last_5pct)
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Dense(64, activation="tanh"))#recomend tanh, or relu
model.add(Dropout(0.1))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
             optimizer=opt,
             metrics = ['accuracy'])
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}')
filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}.hd5" #filename that includes epoch and accuracy
checkpoint = ModelCheckpoint("models\\{}".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, np.array(train_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, np.array(validation_y)),
    callbacks=[tensorboard, checkpoint])

