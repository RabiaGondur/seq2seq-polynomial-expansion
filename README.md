# Sequence-to-Sequence LSTM Model For Polynomial Expansion

## Summary
This project is an implementation of a sequence-to-sequence LSTM model that can expand single variable factorized polynomials. The model was created using the Keras API in Tensorflow, and it was trained using a dataset containing 1 million single variable polynomials. The resulting model after training achieves a high accuracy and low loss when evaluated against a test dataset.

## How to run the project
### Requirements:
This project requires Python3, Tensorflow 2, and NumPy. 
When training and evaluating this model, the following versions were used:

MACOS:
- python 3.8.13
- tensorflow-macos 2.9.2
- numpy 1.22.4

GOOGLE COLAB:
- python 3.7.13
- tensorflow 2.8.2
- numpy 1.21.6

### Training the model:
In order to train the model make sure that the train.txt file is in the working directory and run:
```
python training_code.py
```
This will generate a trained model called TRAINED_model, which can be used for evaluation.

### Evaluating the model:
In order to evaluate the model against a test dataset, first make sure to have a test.txt file in the working directory and then run:
```
python main.py -test
```

## Project Breakdown

### Data Pre-processing
As stated, the dataset being used is comprised of 1 million factorized single variable polynomials as well as their expanded counterparts in a single text file. In order to use this data in the neural network, some pre-processing was required. Before feeding the data into the network, the data had to be turned from its original string format into a numerical representation. This was achieved by encoding the data into one-hot vectors. Both forms of all polynomials were separated and their characters were mapped to numerical values, then those values were encoded into one-hot vectors. After this, the data was ready to be loaded into the model, however, it was also important to consider how the data was loaded into the GPU memory. Due to the high volume of data, it had to be loaded into the GPU memory in batches in order to prevent a GPU memory overload. 


### Model Summary
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 512)               1146880   
                                                                 
 repeat_vector (RepeatVector  (None, 29, 512)          0         
 )                                                               
                                                                 
 lstm_1 (LSTM)               (None, 29, 512)           2099200   
                                                                 
 time_distributed (TimeDistr  (None, 29, 47)           24111     
 ibuted)                                                         
                                                                 
=================================================================
Total params: 3,270,191
Trainable params: 3,270,191
Non-trainable params: 0
_________________________________________________________________
```

### Results


Final accuracy score against a test dataset: ~0.99


