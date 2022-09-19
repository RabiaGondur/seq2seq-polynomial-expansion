import sys
import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #

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
    while len(string2int) != 29:
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

model = keras.models.load_model('TRAINED_model')

count_data_points = 0
def predict(factors: str):
    input = one_hot_encoder(factors)
    predict_x=model.predict(np.array([input]), verbose=0)
    preds=np.argmax(predict_x, axis = 2)
    preds_one_hot = []
    for x in preds[0]:
        current_one_hot = np.zeros((47))
        current_one_hot[x]=1
        preds_one_hot.append(current_one_hot)
    preds_one_hot = np.array(preds_one_hot)
    guess = one_hot_decoder(preds_one_hot)
    global count_data_points
    count_data_points += 1
    print(f'Done: {count_data_points} data points', end='\r')
    return guess.strip()

# --------- END OF IMPLEMENT THIS --------- #

def main(filepath: str):
    factors, expansions = load_file(filepath)
    print('Predicting dataset...')
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print('\nFinal accuracy score:', np.mean(scores))

if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")