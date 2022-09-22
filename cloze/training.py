from model import BiLSTM_new, BiLSTM
import pickle
from utiliz import Dataset
import numpy as np
with open('dataset/example.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

model = BiLSTM()
model.model.fit( [np.array(dataset.X_source[0]), np.array(dataset.X_source[1])],dataset.Y_source, batch_size=32, epochs=1, shuffle=True,
                verbose=True)
model.model.evaluate([np.array(dataset.X_target[0]), np.array(dataset.X_target[1])], dataset.Y_target, verbose=True)

layers = ['char_input', 'word_input', 'char_embedding', 'word_embedding', 'lstm_0', 'lstm_1', 'lstm_2', 'lstm_3',
          'lstm_4', 'lstm_5', 'lstm_6', 'lstm_7', 'lstm_8', 'lstm_9', 'concat_0', 'concat_1', 'concat_2', 'concat_3',
          'concat_4', 'concat_5', 'concat_6', 'concat_7', 'concat_8', 'concat_9', 'reshape_0', 'reshape_1', 'reshape_2',
          'reshape_3', 'reshape_4', 'reshape_5', 'reshape_6', 'reshape_7', 'reshape_8',
          'reshape_9', 'concat', 'lstm', 'dropout_1', 'dense_1', 'dense_2', ]

model_new = BiLSTM_new()
for layer in layers:
    weights = model.model.get_layer(layer).get_weights()
    model_new.model.get_layer(layer).set_weights(weights)

model_new.model.fit([np.array(dataset.X_target[0]), np.array(dataset.X_target[1])], dataset.Y_target, batch_size=32, epochs=1, shuffle=True,
                    verbose=True)
