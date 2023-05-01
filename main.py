import tensorflow as tf
import openl3
import pickle as pkl

import numpy as np
import os
import librosa
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *

def save_date_to_pickle():
  print(2)
  exit(0)
  audio_common_speech = []
  labels_common_speech = []
  emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
  }
  root = 'C:/pythonProject/speech_emotion_recognition_model/RAVDESS'
  ld = os.listdir(root)
  ld = sorted(ld)
  print(ld)
  index = 1
  actor_count = 0
  for i in ld:
    for roots, dirs, files in os.walk(root+'/' + i):
      actor_count += 1
      if actor_count == 9:
        with open(f'C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_emb_{str(index)}.pickle', 'wb') as f_1:
          pkl.dump(audio_common_speech, f_1)
        with open(f'C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_cat_{str(index)}.pickle', 'wb') as f_2:
          pkl.dump(labels_common_speech, f_2)
        audio_common_speech = []
        labels_common_speech = []
        actor_count = 1
        index += 1

      files = sorted(files)
      for file in files:
        #print(file)
        if file.endswith(".wav"):
          print(os.path.join(root, i, file))
          audio, sr = librosa.load(os.path.join(root, i, file))
          emb_list, ts_list = openl3.get_audio_embedding(audio, sr, batch_size=1, content_type="music",
                                                         embedding_size=512, input_repr="mel256")
          audio_common_speech.append(emb_list)
          labels_common_speech.append(emotions[file[6:8]])

  with open(f'C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_emb_{str(index)}.pickle', 'wb') as f_1:
    pkl.dump(audio_common_speech, f_1)
  with open(f'C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_cat_{str(index)}.pickle', 'wb') as f_2:
    pkl.dump(labels_common_speech, f_2)

def merge_pickle():
  audio_common_speech = []
  labels_common_speech = []
  with open('C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_emb_1.pickle', 'rb') as f:
    audio_common_speech += pkl.load(f)
  with open('C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_cat_1.pickle', 'rb') as f:
    labels_common_speech += pkl.load(f)
  with open('C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_emb_2.pickle', 'rb') as f:
    audio_common_speech += pkl.load(f)
  with open('C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_cat_2.pickle', 'rb') as f:
    labels_common_speech += pkl.load(f)
  with open('C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_emb_3.pickle', 'rb') as f:
    audio_common_speech += pkl.load(f)
  with open('C:/pythonProject/speech_emotion_recognition_model/pickle_files/audio_cat_3.pickle', 'rb') as f:
    labels_common_speech += pkl.load(f)

  print('audio_common_speech:\n',audio_common_speech)
  print('labels_common_speech:\n',labels_common_speech)
  return audio_common_speech, labels_common_speech

def get_emb_mean(audio_common_speech, labels_common_speech, observed_emotions):
  new_labels_common_speech = []
  data_x = []
  for f in range(len(audio_common_speech)):
    if labels_common_speech[f] not in observed_emotions:
      continue
    data_x.append(audio_common_speech[f].mean(axis=0))
    new_labels_common_speech.append(labels_common_speech[f])
  return data_x, new_labels_common_speech

def get_lables_categorical(lables_y, observed_emotions):
  i = 0
  categories ={}
  for category in ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful','disgust', 'surprised']:
    if category not in observed_emotions:
      continue
    categories[category] = i
    i+=1
  print("categories ", categories)

  lables_categorical_y = []
  for d in lables_y:
    lables_categorical_y.append(to_categorical(categories[d], num_classes=len(categories)))
  lables_categorical_y = np.asarray(lables_categorical_y)
  print(lables_categorical_y)
  print(lables_categorical_y.shape)
  return lables_categorical_y


def create_model(num_target_classes):

  model = Sequential()
  model.add(Conv1D(filters=256, kernel_size=8, padding='same', activation='relu', input_shape=(512, 1)))
  model.add(Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', input_shape=(512, 1)))
  model.add(Dropout(0.1))
  model.add(MaxPooling1D(pool_size=8))
  model.add(Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', input_shape=(512, 1)))
  model.add(MaxPooling1D(pool_size=8))
  model.add(LSTM(units=128, return_sequences=False, input_shape=(512, 1)))
  model.add(Dense(units=64, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(units=32, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(units=32, activation='relu'))
  model.add(Dense(units=num_target_classes, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
  return model

def show_train_val_loss():
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)

  plt.plot(epochs, loss, 'ro', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def show_train_val_accuracy():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'ro', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()

def scores(data_x, lables_categorical_y,observed_emotions):
  X_test = np.expand_dims(data_x[training_samples + validation_samples:], -1)
  print(X_test.shape)
  y_test = lables_categorical_y[training_samples + validation_samples:]
  y_pred = model_N.predict(X_test)
  y_true = tf.keras.backend.argmax(y_test, axis=-1)
  y_pred = tf.keras.backend.argmax(y_pred, axis=-1)
  print(classification_report(y_true, y_pred, target_names = observed_emotions))
  print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
  observed_emotions = ['happy', 'sad', 'angry', 'surprised']
  num_target_classes = len(observed_emotions)
  audio_common_speech = []
  labels_common_speech = []
  data_x = []
  lables_categorical_y = []

  audio_common_speech, labels_common_speech = merge_pickle()
  data_x,labels_common_speech = get_emb_mean(audio_common_speech,labels_common_speech, observed_emotions)
  data_x = np.asarray(data_x)
  print('data_x\n ', data_x)
  print(data_x.shape)
  lables_categorical_y = get_lables_categorical(labels_common_speech, observed_emotions)
  print('lables_categorical_y.shape: ', lables_categorical_y.shape)
  print('data_x.shape: ', data_x.shape)

  number_of_samples = data_x.shape[0]
  training_samples = int(number_of_samples * 0.8)
  validation_samples = int(number_of_samples * 0.1)
  test_samples = int(number_of_samples * 0.1)
  print('number_of_samples: ', number_of_samples)
  print('training_samples: ', training_samples)
  print('validation_samples: ', validation_samples)
  print('test_samples: ', test_samples)

  model_N = create_model( num_target_classes )


  history = model_N.fit(np.expand_dims(data_x[:training_samples], -1),
                        lables_categorical_y[:training_samples], validation_data=(
    np.expand_dims(data_x[training_samples:training_samples + validation_samples], -1),
    lables_categorical_y[training_samples:training_samples + validation_samples]), epochs=130, shuffle=True)

  model_N.evaluate(np.expand_dims(data_x[training_samples + validation_samples:], -1),
                   lables_categorical_y[training_samples + validation_samples:])

  print(model_N.summary())

  #model_N.save('modelN.h5')
  show_train_val_loss()
  show_train_val_accuracy()
  scores(data_x, lables_categorical_y, observed_emotions)



