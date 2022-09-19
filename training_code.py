import sys
import numpy as np
import random
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense

TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()

    #Shuffling data
    random.shuffle(data)
    open('shuffled_data.txt', 'w').writelines(data)
    shuffled_data = open('shuffled_data.txt', "r").readlines()

    factors, expansions = zip(*[line.strip().split("=") for line in shuffled_data])
    return factors, expansions

MAX_SEQUENCE_LENGTH = 29
ALPHABET = ('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')
NUMERALS = ('0','1','2','3','4','5','6','7','8','9')
SYMBOLS = ('(',')','+','-','/','*','**','sin','cos','tan', ' ')
ALL_CHARS = ALPHABET + NUMERALS + SYMBOLS

# Creating dictionaries to map all possible characters to integers
int2char = dict(enumerate(ALL_CHARS))
char2int = {char: ind for ind, char in int2char.items()}

# Creating one-hot map of all possible characters
one_hot_map = to_categorical(list(int2char.keys()), len(ALL_CHARS))
one_hot_map_lists = []
for x in one_hot_map:
    one_hot_map_lists.append(x.tolist())

def one_hot_encoder(string: str):
    """ A helper function that encodes a string to an array of one-hot vectors 

    :param string: string to be encoded
    :return array of one-hot vectors: numpy array of one-hot vectors corresponding to padded string
    """
    string_iter = iter(range(len(string)))
    string2int = []
    for i in string_iter:
        # Catching symbols with multiple characters
        try:
            if string[i] == '*' and string[i+1]=='*':
                string2int.append(42) # 42 corresponds to '**' in the map
                next(string_iter, None)
                continue
            elif string[i] == 's' and string[i+1]=='i' and string[i+2]=='n':
                string2int.append(43) # 43 corresponds to 'sin' in the map
                next(string_iter, None)
                next(string_iter, None)
                continue
            elif string[i] == 'c' and string[i+1]=='o' and string[i+2]=='s': 
                string2int.append(44) # 44 corresponds to 'cos' in the map
                next(string_iter, None)
                next(string_iter, None)
                continue
            elif string[i] == 't' and string[i+1]=='a' and string[i+2]=='n': 
                string2int.append(45) # 45 corresponds to 'tan' in the map
                next(string_iter, None)
                next(string_iter, None)
                continue
            else:
                for int, char in int2char.items():
                    if string[i] == char:
                        string2int.append(int)
                        break
        except:
            for int, char in int2char.items():
                    if string[i] == char:
                        string2int.append(int)
                        break
    # Padding the sequences so it reaches MAX_SEQUENCE_LENGTH
    while len(string2int) != MAX_SEQUENCE_LENGTH:
        string2int.append(46) # 46 corresponds to ' ' in the map
    one_hot = []
    for int in string2int:
        one_hot.append(one_hot_map[int, :])
    return np.array(one_hot)

def one_hot_decoder(one_hot):
    """ A helper function that decodes an array of one-hot vectors to a string

    :param one_hot: array of one-vectors to be decoded
    :return string: corresponding string with padding
    """
    translated = ''
    for x in one_hot:
        translated = translated + int2char[one_hot_map_lists.index(x.tolist())]
    return translated

class DataGenerator(Sequence):
    """ A helper class that generates data for the model in batches in order to avoid GPU memory leaks
    """
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def main(filepath: str):
    factors, expansions = load_file(filepath)
    print("Encoding data to one-hot vectors...")
    print('----------------------')
    factored_eqs = []
    for x in factors:
        factored_eqs.append(one_hot_encoder(x))
        print(f'Encoding factored set: {len(factored_eqs)}/{len(factors)}', end='\r')
    factored_eqs = np.array(factored_eqs)
    print('\n----------------------')
    expanded_eqs = []
    for x in expansions:
        expanded_eqs.append(one_hot_encoder(x))
        print(f'Encoding expanded set: {len(expanded_eqs)}/{len(expansions)}', end='\r')
    expanded_eqs = np.array(expanded_eqs)
    print('\n----------------------')
    
    # The code below takes 1% of the dataset in order to create a small validation dataset
    split_at = len(factored_eqs) - len(factored_eqs) // 100
    (x_train, x_val) = factored_eqs[:split_at], factored_eqs[split_at:]
    (y_train, y_val) = expanded_eqs[:split_at], expanded_eqs[split_at:]

    hidden_size = 512
    batch_size = 1024

    # NEURAL NET
    optimizer = keras.optimizers.Adam(lr=0.001)
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(MAX_SEQUENCE_LENGTH, 47)))
    model.add(RepeatVector(MAX_SEQUENCE_LENGTH))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(47, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    model.summary()

    train_gen = DataGenerator(x_train, y_train, batch_size)
    for epoch in range(1, 60):
        print()
        print('-' * 50)
        print('Epoch', epoch)
        model.fit(train_gen,
                epochs=1)  
        # After each epoch 10 samples from the validation set will be selected at random to visualize performance of model
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            predict_x=model.predict(rowx, verbose=0) 
            preds=np.argmax(predict_x, axis = 2)
            preds_one_hot = []
            for x in preds[0]:
                current_one_hot = np.zeros((47))
                current_one_hot[x]=1
                preds_one_hot.append(current_one_hot)
            preds_one_hot = np.array(preds_one_hot)
            q = one_hot_decoder(rowx[0])
            correct = one_hot_decoder(rowy[0])
            guess = one_hot_decoder(preds_one_hot)
            print('F', q, end=' ')
            print('E', correct, end=' ')
            if correct == guess:
                print('☑', end=' ')
            else:
                print('☒', end=' ')
            print(guess)
    model.save('mymodel_FINAL')    

if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")