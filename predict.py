from main import *
import tensorflow as tf


model_N = tf.keras.models.load_model('model_conv_lstm_4.h5')

emotions = { 1: 'happy', 2: 'sad', 3: 'angry', 4: 'surprised'}

def predict(wav_filepath):
  emb_data = []
  audio, sr = librosa.load(wav_filepath)
  audio_emb_list, ts_list = openl3.get_audio_embedding(audio, sr, batch_size=1, content_type="music",
                                                       embedding_size=512, input_repr="mel256")

  emb_data = np.mean(audio_emb_list, axis=0)
  emb_data_array = np.asarray(emb_data)
  emb_data_array = np.reshape(emb_data_array, newshape= (1, 512, 1))
  predictions=model_N.predict(emb_data_array)
  print(emotions[np.argmax(predictions[0])+1])
  return emotions[np.argmax(predictions[0])+1]


