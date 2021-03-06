import csv
import os.path
import shutil
from os import path
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa.display
import splitfolders
import tensorflow as tf
from PIL import Image
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, roc_curve
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from pandas.core.common import random_state
from pydub import AudioSegment
from pydub.silence import split_on_silence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

disease = 'Asma InSalute'.split()
csvname = 'Asma_Android InSalute_Android Asma_IOS InSalute_IOS'.split()

"""CSV PARSING"""
if not path.exists('./Audio'):
    for nf in csvname:
        print(nf)
        p = []
        fn = []
        b = []
        j = 0
        p_file = f"{nf}.csv"
        data = pd.read_csv(p_file, sep=';', on_bad_lines='skip', usecols=['Uid', 'FolderName', 'Breathfilename'])
        for patient in data.Uid:
            p.append(patient)
        for folder in data.FolderName:
            fn.append(folder)
        for bfile in data.Breathfilename:
            b.append(bfile)
        if nf == 'InSalute_Android':
            fn_asma = []
            asma_file = f"Asma_Android.csv"
            data_asma = pd.read_csv(asma_file, sep=';', on_bad_lines='skip', usecols=['Uid', 'FolderName'])
            for data in data_asma.FolderName:
                fn_asma.append(data)
            num = len(fn_asma)
            fn = fn[0:num]
        if nf == 'InSalute_IOS':
            fn_asma = []
            asma_file = f"Asma_IOS.csv"
            data_asma = pd.read_csv(asma_file, sep=';', on_bad_lines='skip', usecols=['Uid', 'FolderName'])
            for data in data_asma.FolderName:
                fn_asma.append(data)
            num = len(fn_asma)
            fn = fn[0:num]
        for index in range(len(fn)):
            if nf == 'InSalute_IOS' or nf == 'Asma_IOS':
                j = j + 1
                d = nf.rpartition('_')[0]
                filename = b[index]
                f = filename[:-3].replace(".", ".wav")
                pathlib.Path(f'./Audio/{d}').mkdir(parents=True, exist_ok=True)
                src_name = f'./covid19_data_0426/{p[index]}/{fn[index]}/{f}'
                f2 = filename[:-3].replace(".", f"{j}.wav")
                dst_name = f'./Audio/{d}/{f2}'
                shutil.copy(src_name, dst_name)
            else:
                d = nf.rpartition('_')[0]
                pathlib.Path(f'./Audio/{d}').mkdir(parents=True, exist_ok=True)
                src_name = f'./covid19_data_0426/{p[index]}/{fn[index]}/{b[index]}'
                dst_name = f'./Audio/{d}/'
                shutil.copy(src_name, dst_name)

"""if not path.exists('./images'):
    for d in disease:
        pathlib.Path(f'./images/{d}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'./Audio/{d}'):
            audio = f'./Audio/{d}/{filename}'
            aud, Fs = librosa.load(audio, mono=True, duration=125)
            powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
            plt.axis('off')
            plt.savefig(f'images/{d}/{filename[:-3].replace(".", "")}.png')
            plt.clf()"""

"""if not path.exists('./images'):
    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(8,8))
    for d in disease:
        pathlib.Path(f'images/{d}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'./Audio/{d}'):
            fname = f'./Audio/{d}/{filename}'
            y, sr = librosa.load(fname, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off')
            plt.savefig(f'images/{d}/{filename[:-3].replace(".", "")}.png')
            plt.clf()"""
"""print("inizio la lavorazione del nuovo csv con le feature")
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
for d in disease:
    for filename in os.listdir(f'./Audio/{d}'):
        fname = f'./Audio/{d}/{filename}'
        y, sr = librosa.load(fname, mono=True, duration=30)
        rmse = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {d}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())"""

data = pd.read_csv('dataset.csv')
data.head()  # Dropping unneccesary columns
data = data.drop(['filename'], axis=1)
# Encoding the Labels
disease_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(disease_list)
# Scaling the Feature columns
# scaler = StandardScaler()
X = np.array(data.iloc[:, :-1], dtype=float)
# Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(y_test))
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""model=Sequential()  #loss 0,40   accuracy 0,59
###first layer
model.add(Dense(100,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(1))
model.add(Activation('sigmoid'))"""

"""model = Sequential()   0,38 loss    0,61 accuracy     con test 0,3 loss 0,42    accuracy 0,57

model.add(Dense(186, input_shape=(X_train.shape[1],), activation = 'relu'))

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.6))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))"""

"""model = Sequential()    #0,40 loss 0,59 accuracy
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))"""

"""model = Sequential()    #loss 0,37    accuracy 0,62   con 0,3 test_size   loss 0,41   accuracy 0,58  | con 200 epoche loss 0,35  accuracy 0,64
model.add(Dense(256, kernel_initializer='random_uniform', activation = 'relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer='random_uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='random_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='random_uniform', activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)"""

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=1)

scores = model.evaluate(X_train, y_train, verbose=1)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=1)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# Results - Confusion Matrix
y_test_pred = model.predict(X_test)
y_test_pred = (y_test_pred > 0.5)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Asma', 'In salute'], yticklabels=['Asma', 'In salute'], cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()

y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
plt.clf()

"""y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)"""



