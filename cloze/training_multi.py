from model import BiLSTM_new, BiLSTM, BiLSTM_relation, BiLSTM_relation_new
import pickle
from utiliz import Dataset
import numpy as np

with open('dataset/example.pkl', 'rb') as fp:
    dataset = pickle.load(fp)
with open('dataset/example_relation.pkl', 'rb') as fp:
    dataset_relation = pickle.load(fp)

EPOCH = 1

model = BiLSTM()
model_relation = BiLSTM_relation()

shared_layers = ['char_embedding', 'word_embedding', 'lstm_0', 'lstm_1', 'lstm_2', 'lstm_3',
                 'lstm_4', 'lstm_5', 'lstm_6', 'lstm_7', 'lstm_8', 'lstm_9', 'concat_0', 'concat_1', 'concat_2',
                 'concat_3', 'concat_4', 'concat_5', 'concat_6', 'concat_7', 'concat_8', 'concat_9', 'reshape_0',
                 'reshape_1', 'reshape_2', 'reshape_3', 'reshape_4', 'reshape_5', 'reshape_6', 'reshape_7', 'reshape_8',
                 'reshape_9']
for ep in range(EPOCH):
    model.model.fit([np.array(dataset.X_source[0]), np.array(dataset.X_source[1])], dataset.Y_source, batch_size=32,
                    epochs=1, shuffle=True,
                    verbose=True)
    # model.model.evaluate([np.array(dataset.X_target[0]), np.array(dataset.X_target[1])], dataset.Y_target, verbose=True)
    for layer in shared_layers:
        weights = model.model.get_layer(layer).get_weights()
        model_relation.model.get_layer('sub_'+layer).set_weights(weights)
        model_relation.model.get_layer('obj_' + layer).set_weights(weights)
    model_relation.model.fit([
        np.array([data[0] for data in dataset_relation.X_source[0]]),
        np.array([data[0] for data in dataset_relation.X_source[1]]),
        np.array([data[1] for data in dataset_relation.X_source[0]]),
        np.array([data[1] for data in dataset_relation.X_source[1]])],
        dataset_relation.Y_source, batch_size=32, epochs=1, shuffle=True, verbose=True)

'''
-------------------------------------------------- for new model ---------------------------------------------
'''
layers = ['char_input', 'word_input', 'char_embedding', 'word_embedding', 'lstm_0', 'lstm_1', 'lstm_2', 'lstm_3',
          'lstm_4', 'lstm_5', 'lstm_6', 'lstm_7', 'lstm_8', 'lstm_9', 'concat_0', 'concat_1', 'concat_2', 'concat_3',
          'concat_4', 'concat_5', 'concat_6', 'concat_7', 'concat_8', 'concat_9', 'reshape_0', 'reshape_1', 'reshape_2',
          'reshape_3', 'reshape_4', 'reshape_5', 'reshape_6', 'reshape_7', 'reshape_8',
          'reshape_9', 'concat', 'lstm', 'dropout_1', 'dense_1', 'dense_2', ]

layers_relation = ['sub_char_input', 'obj_char_input', 'sub_word_input', 'sub_char_embedding', 'obj_word_input',
                   'obj_char_embedding', 'sub_word_embedding', 'obj_word_embedding', 'sub_lstm_0', 'sub_lstm_1',
                   'sub_lstm_2', 'sub_lstm_3', 'sub_lstm_4', 'sub_lstm_5', 'sub_lstm_6', 'sub_lstm_7', 'sub_lstm_8',
                   'sub_lstm_9', 'obj_lstm_0', 'obj_lstm_1', 'obj_lstm_2', 'obj_lstm_3', 'obj_lstm_4', 'obj_lstm_5',
                   'obj_lstm_6', 'obj_lstm_7', 'obj_lstm_8', 'obj_lstm_9', 'sub_concat_0', 'sub_concat_1',
                   'sub_concat_2', 'sub_concat_3', 'sub_concat_4', 'sub_concat_5', 'sub_concat_6', 'sub_concat_7',
                   'sub_concat_8', 'sub_concat_9', 'obj_concat_0', 'obj_concat_1', 'obj_concat_2', 'obj_concat_3',
                   'obj_concat_4', 'obj_concat_5', 'obj_concat_6', 'obj_concat_7', 'obj_concat_8', 'obj_concat_9',
                   'sub_reshape_0', 'sub_reshape_1', 'sub_reshape_2', 'sub_reshape_3', 'sub_reshape_4', 'sub_reshape_5',
                   'sub_reshape_6', 'sub_reshape_7', 'sub_reshape_8', 'sub_reshape_9', 'obj_reshape_0', 'obj_reshape_1',
                   'obj_reshape_2', 'obj_reshape_3', 'obj_reshape_4', 'obj_reshape_5', 'obj_reshape_6', 'obj_reshape_7',
                   'obj_reshape_8', 'obj_reshape_9', 'concat', 'lstm', 'dropout_1', 'dense_1', 'dense_2', 'main_output']

model_new = BiLSTM_new()
model_relation_new = BiLSTM_relation_new()

for layer in layers:
    weights = model.model.get_layer(layer).get_weights()
    model_new.model.get_layer(layer).set_weights(weights)

for layer in layers_relation:
    weights = model_relation.model.get_layer(layer).get_weights()
    model_relation_new.model.get_layer(layer).set_weights(weights)

for ep in range(EPOCH):
    model_new.model.fit([np.array(dataset.X_target[0]), np.array(dataset.X_target[1])], dataset.Y_target, batch_size=32, epochs=1, shuffle=True,
                        verbose=True)
    for layer in shared_layers:
        weights = model_new.model.get_layer(layer).get_weights()
        model_relation_new.model.get_layer('sub_'+layer).set_weights(weights)
        model_relation_new.model.get_layer('obj_'+ layer).set_weights(weights)

    model_relation_new.model.fit([
        np.array([data[0] for data in dataset_relation.X_target[0]]),
        np.array([data[0] for data in dataset_relation.X_target[1]]),
        np.array([data[1] for data in dataset_relation.X_target[0]]),
        np.array([data[1] for data in dataset_relation.X_target[1]])],
        dataset_relation.Y_target, batch_size=32, epochs=1, shuffle=True, verbose=True)
