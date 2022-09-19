from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Input, Concatenate, Reshape
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras.backend as K

class BiLSTM():

    def __init__(self):
        self.word_maxlen = 2
        self.word_maxchar = 6
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
            word_embedding_list.append(word_embedding[:,i,:])
            lstm_char = Bidirectional(LSTM(64))(char_embedding[:, i * self.word_maxchar:(i + 1) * self.word_maxchar, :])
            # char_embedding_list.append(lstm_char)
            embedding_list.append(Reshape((1,32+128))(Concatenate(axis=1)([word_embedding[:,i,:], lstm_char])))
        final_embedding = Concatenate(axis=1)([embed for embed in embedding_list])
        lstm_1 = Dropout(0.2)(Bidirectional(LSTM(64))(final_embedding))
        dense_1 = Dense(128, activation='relu',name='dense_1')(lstm_1)
        dense_2 = Dense(64, activation='relu',name='dense_2')(dense_1)
        output = Dense(self.output_classes, activation='softmax', name='main_output')(dense_2)
        # char_embedding = K.variable(value = [[1,2,3,4]])
        # final_word_embedding = Concatenate(axis=1)([word_embedding_list[5], char_embedding])
        model = Model(inputs=[word_input, char_input], outputs=output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        # word_input = Input(shape=self.word_maxlen, name='word_input')
        # word_embedding = Embedding(output_dim=512, input_dim=self.word_vocaburaly_size, input_length=self.word_maxlen, name='word_embedding')(word_input)
        # char_input = Input(shape=self.char_maxlen, name='char_input')
        # char_embedding = Embedding(output_dim=512, input_dim=self.char_vocaburaly_size, input_length=self.char_maxlen, name='char_embedding')(char_input)
        # # embed_output = Concatenate(axis=1)([word_embedding, char_embedding])
        # lstm_1 = Bidirectional(LSTM(64))(word_embedding)
        # lstm_2 = Bidirectional(LSTM(64))(char_embedding)
        # lstm_out = Concatenate(axis=1)([lstm_1, lstm_2])
        # dropout_out = Dropout(0.2)(lstm_out)
        # output = Dense(self.output_classes, activation='softmax', name='main_output')(dropout_out)
        # model = Model(inputs=[word_input, char_input], outputs=output)
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        print(model.summary())
        return model
# word_1 = [[1,2]]
# char_1 = [[1,2,3,3,2,1,1,2,3,3,2,1]]
# model = BiLSTM()
# import numpy as np
# print(model.model.predict([np.array(word_1), np.array(char_1)]).shape)
