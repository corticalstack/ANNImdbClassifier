from keras.datasets import imdb
from keras import models, layers, optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt


class ImdbClassifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

        self.load_data()

        self.decode_review()

        # Encode integer sequences into binary matrix
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)

        # Vectorize labels
        self.y_train = np.asarray(self.train_labels).astype('float32')
        self.y_test = np.asarray(self.test_labels).astype('float32')

        self.x_val = self.x_train[:10000]
        self.partial_x_train = self.x_train[10000:]

        self.y_val = self.y_train[:10000]
        self.partial_y_train = self.y_train[10000:]

        self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(self.partial_x_train, self.partial_y_train, epochs=20, batch_size=512,
                            validation_data=(self.x_val, self.y_val))

        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def load_data(self):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words=10000)
        print('Train data shape', self.train_data.shape)
        print('Train labels shape', self.train_labels.shape)
        print('Test data shape', self.test_data.shape)
        print('Test labels shape', self.test_labels.shape)

    def decode_review(self):
        word_index = imdb.get_word_index()  # dictionary mapping words to integer index

        # Map integer indices to words
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        # Indices offset by 3, as 0, 1 and 2 reserved for padding, start of sequence and unknown
        decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in self.train_data[0]])
        print(decoded_review)

    # Manual implementation of 1-hot encoding to transform list into vectors of 0's and 1's
    @staticmethod
    def vectorize_sequences(sequences, dimension=10000):

        # Create all-zero matrix of shape (len(sequences, dimension))
        results = np.zeros((len(sequences), dimension))

        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1  # Set specific indices of results[i] to 1
        return results

imdbclassifier = ImdbClassifier()