from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Input, Concatenate, Reshape
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras.backend as K


class BiLSTM():

    def __init__(self):
        self.word_maxlen = 10
        self.word_maxchar = 10
        self.char_maxlen = self.word_maxlen * self.word_maxchar
        self.word_vocab_size = 720
        self.char_vocab_size = 27
        self.output_classes = 460

        self.model = self.building_model()

    def building_model(self):
        word_input = Input(shape=self.word_maxlen, name='word_input')
        word_embedding = Embedding(output_dim=32, input_dim=self.word_vocab_size, input_length=self.word_maxlen,
                                   name='word_embedding')(word_input)
        char_input = Input(shape=self.char_maxlen, name='char_input')
        char_embedding = Embedding(output_dim=32, input_dim=self.char_vocab_size, input_length=self.char_maxlen,
                                   name='char_embedding')(char_input)
        word_embedding_list = []
        embedding_list = []
        for i in range(self.word_maxlen):
            word_embedding_list.append(word_embedding[:, i, :])
            lstm_char = Bidirectional(LSTM(64), name='lstm_' + str(i))(
                char_embedding[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            embedding_list.append(Reshape((1, 32 + 128), name='reshape_' + str(i))(
                Concatenate(axis=1, name='concat_' + str(i))([word_embedding[:, i, :], lstm_char])))
        final_embedding = Concatenate(axis=1, name='concat')([embed for embed in embedding_list])
        lstm_1 = Dropout(0.2, name='dropout_1')(Bidirectional(LSTM(64), name='lstm')(final_embedding))
        dense_1 = Dense(128, activation='relu', name='dense_1')(lstm_1)
        dense_2 = Dense(64, activation='relu', name='dense_2')(dense_1)
        output = Dense(self.output_classes, activation='softmax', name='main_output')(dense_2)

        model = Model(inputs=[word_input, char_input], outputs=output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        return model


class BiLSTM_new():

    def __init__(self):
        self.word_maxlen = 10
        self.word_maxchar = 10
        self.char_maxlen = self.word_maxlen * self.word_maxchar
        self.word_vocab_size = 720
        self.char_vocab_size = 27
        self.output_classes = 460

        self.model = self.building_model()

    def building_model(self):
        word_input = Input(shape=self.word_maxlen, name='word_input')
        word_embedding = Embedding(output_dim=32, input_dim=self.word_vocab_size, input_length=self.word_maxlen,
                                   name='word_embedding')(word_input)
        char_input = Input(shape=self.char_maxlen, name='char_input')
        char_embedding = Embedding(output_dim=32, input_dim=self.char_vocab_size, input_length=self.char_maxlen,
                                   name='char_embedding')(char_input)
        word_embedding_list = []
        embedding_list = []
        for i in range(self.word_maxlen):
            word_embedding_list.append(word_embedding[:, i, :])
            lstm_char = Bidirectional(LSTM(64), name='lstm_' + str(i))(
                char_embedding[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            embedding_list.append(Reshape((1, 32 + 128), name='reshape_' + str(i))(
                Concatenate(axis=1, name='concat_' + str(i))([word_embedding[:, i, :], lstm_char])))
        final_embedding = Concatenate(axis=1, name='concat')([embed for embed in embedding_list])
        lstm_1 = Dropout(0.2, name='dropout_1')(Bidirectional(LSTM(64), name='lstm')(final_embedding))
        dense_1 = Dense(128, activation='relu', name='dense_1')(lstm_1)
        dense_2 = Dense(64, activation='relu', name='dense_2')(dense_1)
        reinput = Reshape((64, 1))(dense_2)
        lstm_2 = Dropout(0.2)(Bidirectional(LSTM(64))(reinput))
        dense_3 = Dense(128, activation='relu', name='dense_3')(lstm_2)
        dense_4 = Dense(64, activation='relu', name='dense_4')(dense_3)
        output = Dense(self.output_classes, activation='softmax', name='main_output')(dense_4)

        model = Model(inputs=[word_input, char_input], outputs=output)
        model.summary()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        return model


class BiLSTM_relation():

    def __init__(self):
        self.word_maxlen = 10
        self.word_maxchar = 10
        self.char_maxlen = self.word_maxlen * self.word_maxchar
        self.word_vocab_size = 720
        self.char_vocab_size = 27
        self.output_classes = 10

        self.model = self.building_model()

    def building_model(self):
        word_input_sub = Input(shape=self.word_maxlen, name='sub_word_input')
        word_input_obj = Input(shape=self.word_maxlen, name='obj_word_input')
        word_embedding_sub = Embedding(output_dim=32, input_dim=self.word_vocab_size, input_length=self.word_maxlen,
                                       name='sub_word_embedding')(word_input_sub)
        word_embedding_obj = Embedding(output_dim=32, input_dim=self.word_vocab_size, input_length=self.word_maxlen,
                                       name='obj_word_embedding')(word_input_obj)

        char_input_sub = Input(shape=self.char_maxlen, name='sub_char_input')
        char_input_obj = Input(shape=self.char_maxlen, name='obj_char_input')
        char_embedding_sub = Embedding(output_dim=32, input_dim=self.char_vocab_size, input_length=self.char_maxlen,
                                       name='sub_char_embedding')(char_input_sub)
        char_embedding_obj = Embedding(output_dim=32, input_dim=self.char_vocab_size, input_length=self.char_maxlen,
                                       name='obj_char_embedding')(char_input_obj)

        word_embedding_list_sub = []
        embedding_list_sub = []
        for i in range(self.word_maxlen):
            word_embedding_list_sub.append(word_embedding_sub[:, i, :])
            lstm_char_sub = Bidirectional(LSTM(64), name='sub_lstm_' + str(i))(
                char_embedding_sub[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            embedding_list_sub.append(Reshape((1, 32 + 128), name='sub_reshape_' + str(i))(
                Concatenate(axis=1, name='sub_concat_' + str(i))([word_embedding_sub[:, i, :], lstm_char_sub])))

        word_embedding_list_obj = []
        embedding_list_obj = []
        for i in range(self.word_maxlen):
            word_embedding_list_obj.append(word_embedding_obj[:, i, :])
            lstm_char_obj = Bidirectional(LSTM(64), name='obj_lstm_' + str(i))(
                char_embedding_obj[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            embedding_list_obj.append(Reshape((1, 32 + 128), name='obj_reshape_' + str(i))(
                Concatenate(axis=1, name='obj_concat_' + str(i))([word_embedding_obj[:, i, :], lstm_char_obj])))
        print([embed for embed in embedding_list_sub] + [embed for embed in embedding_list_obj])
        final_embedding = Concatenate(axis=1, name='concat')(
            [embed for embed in embedding_list_sub]+[embed for embed in embedding_list_obj])

        lstm_1 = Dropout(0.2, name='dropout_1')(Bidirectional(LSTM(64), name='lstm')(final_embedding))
        dense_1 = Dense(128, activation='relu', name='dense_1')(lstm_1)
        dense_2 = Dense(64, activation='relu', name='dense_2')(dense_1)
        output = Dense(self.output_classes, activation='softmax', name='main_output')(dense_2)

        model = Model(inputs=[word_input_sub, char_input_sub, word_input_obj, char_input_obj], outputs=output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        return model


class BiLSTM_relation_new():

    def __init__(self):
        self.word_maxlen = 10
        self.word_maxchar = 10
        self.char_maxlen = self.word_maxlen * self.word_maxchar
        self.word_vocab_size = 720
        self.char_vocab_size = 27
        self.output_classes = 10

        self.model = self.building_model()

    def building_model(self):
        word_input_sub = Input(shape=self.word_maxlen, name='sub_word_input')
        word_input_obj = Input(shape=self.word_maxlen, name='obj_word_input')
        word_embedding_sub = Embedding(output_dim=32, input_dim=self.word_vocab_size, input_length=self.word_maxlen,
                                       name='sub_word_embedding')(word_input_sub)
        word_embedding_obj = Embedding(output_dim=32, input_dim=self.word_vocab_size, input_length=self.word_maxlen,
                                       name='obj_word_embedding')(word_input_obj)

        char_input_sub = Input(shape=self.char_maxlen, name='sub_char_input')
        char_input_obj = Input(shape=self.char_maxlen, name='obj_char_input')
        char_embedding_sub = Embedding(output_dim=32, input_dim=self.char_vocab_size, input_length=self.char_maxlen,
                                       name='sub_char_embedding')(char_input_sub)
        char_embedding_obj = Embedding(output_dim=32, input_dim=self.char_vocab_size, input_length=self.char_maxlen,
                                       name='obj_char_embedding')(char_input_obj)

        word_embedding_list_sub = []
        embedding_list_sub = []
        for i in range(self.word_maxlen):
            word_embedding_list_sub.append(word_embedding_sub[:, i, :])
            lstm_char_sub = Bidirectional(LSTM(64), name='sub_lstm_' + str(i))(
                char_embedding_sub[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            embedding_list_sub.append(Reshape((1, 32 + 128), name='sub_reshape_' + str(i))(
                Concatenate(axis=1, name='sub_concat_' + str(i))([word_embedding_sub[:, i, :], lstm_char_sub])))

        word_embedding_list_obj = []
        embedding_list_obj = []
        for i in range(self.word_maxlen):
            word_embedding_list_obj.append(word_embedding_obj[:, i, :])
            lstm_char_obj = Bidirectional(LSTM(64), name='obj_lstm_' + str(i))(
                char_embedding_obj[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            embedding_list_obj.append(Reshape((1, 32 + 128), name='obj_reshape_' + str(i))(
                Concatenate(axis=1, name='obj_concat_' + str(i))([word_embedding_obj[:, i, :], lstm_char_obj])))

        final_embedding = Concatenate(axis=1, name='concat')(
            [embed for embed in embedding_list_sub] + [embed for embed in embedding_list_obj])

        lstm_1 = Dropout(0.2, name='dropout_1')(Bidirectional(LSTM(64), name='lstm')(final_embedding))
        dense_1 = Dense(128, activation='relu', name='dense_1')(lstm_1)
        dense_2 = Dense(64, activation='relu', name='dense_2')(dense_1)
        reinput = Reshape((64, 1))(dense_2)
        lstm_2 = Dropout(0.2, name='dropout_2')(Bidirectional(LSTM(64))(reinput))
        dense_3 = Dense(128, activation='relu', name='dense_3')(lstm_2)
        dense_4 = Dense(64, activation='relu', name='dense_4')(dense_3)
        output = Dense(self.output_classes, activation='softmax', name='main_output')(dense_4)

        model = Model(inputs=[word_input_sub, char_input_sub, word_input_obj, char_input_obj], outputs=output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        return model
# word_1 = [[1,2]]
# char_1 = [[1,2,3,3,2,1,1,2,3,3,2,1]]
# model = BiLSTM()
# import numpy as np
# print(model.model.predict([np.array(word_1), np.array(char_1)]).shape)
