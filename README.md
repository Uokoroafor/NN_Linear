
# Linear Regression with Neural Networks

Code adapted from coursework on Neural Networks for the Introduction to Machine Learning Module.

### house_value_regression.py: 

This contains a Regressor object constructed for linear regression on the California Housing data. It was built with PyTorch, Pandas and  Numpy. This pre-processes data, implements our neural network regressor,  and also evaluates the performance of the regressor given set hyperparameters.


## Run Locally

Clone the project

```bash
  git clone https://github.com/Uokoroafor/NN_Linear.git
  
```

Install dependencies 
```bash
    $ pip install -r requirements.txt
```



## Usage/Examples

```python
from house_value_regression import*

# Given training data (X_train, Y_train) and test data (X_test, Y_test)

# To instantiate the regressor, update the values below as follws:

# X_train - features to pre-process and fit over
# nb_epoch - number of epochs
# lr - the learning rate
# batch_size - the number of observations for the batch gradient descent
# layers - the number of hidden layers in the neural network
# neurons - the number of neurons in each hidden layer
# activation - the activation function for all but the last hidden layer. 
    #   One of ['relu', 'sigmoid', 'tanh']
# output_activation - the activation function for the last hidden layer if desired to be different. 
    #   One of ['relu', 'sigmoid', 'tanh']

 Regressor(X_train, nb_epoch=500, lr=1e-3, batch_size=32, activation='relu', 
 layers=4, neurons=64,output_activation='relu')

# To fit training data
regressor1.fit(X_train, Y_train)

# To calculate prediction error
error1 = regressor.score(X_test, Y_test)

```

