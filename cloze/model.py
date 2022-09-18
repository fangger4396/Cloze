from keras import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Input, Concatenate
from keras.preprocessing import sequence

class BiLSTM():

    def __init__(self):
        self.word_maxlen = 10
        self.char_maxlen = 49
        self.word_vocab_size = 720
        self.char_vocab_size = 27
        self.output_classes = 460

        self.model = self.building_model()

    def building_model(self):
        word_input = Input(shape=self.word_maxlen, name='word_input')
        word_embedding = Embedding(output_dim=512, input_dim=self.word_vocaburaly_size, input_length=self.word_maxlen, name='word_embedding')(word_input)
        char_input = Input(shape=self.char_maxlen, name='char_input')
        char_embedding = Embedding(output_dim=512, input_dim=self.char_vocaburaly_size, input_length=self.char_maxlen, name='char_embedding')(char_input)
        # embed_output = Concatenate(axis=1)([word_embedding, char_embedding])
        lstm_1 = Bidirectional(LSTM(64))(word_embedding)
        lstm_2 = Bidirectional(LSTM(64))(char_embedding)
        lstm_out = Concatenate(axis=1)([lstm_1, lstm_2])
        dropout_out = Dropout(0.2)(lstm_out)
        output = Dense(self.output_classes, activation='softmax', name='main_output')(dropout_out)
        model = Model(inputs=[word_input, char_input], outputs=output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        print(model.summary())
        return model