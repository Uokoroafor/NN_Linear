import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class Regressor:

    def __init__(self, x, nb_epoch=500, lr=1e-3, batch_size=32, activation='relu', output_activation='relu', layers=4,
                 neurons=64):
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        self.nb_epoch = nb_epoch
        self.num_vars = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                         'households', 'median_income']
        self.cat_var = ['ocean_proximity']
        self.target_var = ['median_house_value']
        self.encoder = None
        self.y_scaler = None
        self.scaler = None
        self.learning_rate = lr
        self.batch_size = batch_size

        self.error_fn = None
        self.optim = None
        self.model = None

        self.output_size = 1
        self.errors = []
        self.activation = activation
        self.output_activation = output_activation
        self.layers = layers
        self.neurons = neurons
        self.losses = []
        self.x = x
        X, _ = self._preprocessor(x=self.x, training=True)
        self.input_size = X.shape[1]
        return

    class Model(nn.Module):
        def __init__(self, input_size, activation, layers, neurons, output_activation='relu'):
            super().__init__()
            self.input_size = input_size
            self.layers = layers
            self.neurons = neurons
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sig':
                self.activation = nn.Sigmoid()
            elif activation == 'tanh':
                self.activation = nn.Tanh()

            if output_activation == 'sig':
                self.output_activation = nn.Sigmoid()
            elif output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            elif output_activation == 'relu':
                self.output_activation = nn.ReLU()

            # define the input layer
            self.input_layer = nn.Linear(in_features=self.input_size, out_features=self.neurons)
            if self.layers > 0:
                inner_layers = [nn.Linear(in_features=self.neurons, out_features=self.neurons) for k in
                                range(self.layers)]
                self.inner_layers = nn.ModuleList(inner_layers)
            else:
                self.inner_layers = None

            # Output Layer has to be linear as we are in linear regression
            self.output_layer = nn.Linear(in_features=self.neurons, out_features=1)

            # Initialise weights randomly and set biases to zero

            for layer in self.inner_layers:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in [self.output_layer, self.input_layer]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

            self.nn_model = self.apply_layers()

        def apply_layers(self):
            # input is reshaped to the 1D array and fed into the input layer
            # input = input.reshape((input.size(0), self.input_size, 1))

            # Activation function is applied to the input layer
            layer_list = [self.input_layer]

            if self.inner_layers:
                for layer in self.inner_layers:
                    layer_list.append(self.activation)
                    layer_list.append(layer)

            # output layer is applied
            layer_list.append(self.output_activation)
            layer_list.append(self.output_layer)

            return nn.Sequential(*layer_list)

        def forward(self, input):

            return self.nn_model(input)

    def get_params(self, deep=True):
        params = {'x': self.x, 'nb_epoch': self.nb_epoch, 'lr': self.learning_rate, 'layers': self.layers,
                  'batch_size': self.batch_size, 'neurons': self.neurons,
                  'activation': self.activation, 'output_activation': self.output_activation}
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        ##### Reset the model #####
        self.num_vars = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                         'households', 'median_income']
        self.cat_var = ['ocean_proximity']
        self.target_var = ['median_house_value']
        self.encoder = None
        self.y_scaler = None
        self.scaler = None

        self.error_fn = None
        self.optim = None
        self.model = None

        self.output_size = 1
        self.errors = []
        self.losses = []
        X, _ = self._preprocessor(x=self.x, training=True)
        self.input_size = X.shape[1]
        return self

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} -- Preprocessed target array of
              size (batch_size, 1).

        """
        # First we replace NAs and Infs
        x1 = x.copy(deep=True)
        x1 = self.replace_nas(x1)
        x1 = self.replace_infs(x1)

        # Encode the categorical variables
        x1_cat = self.encode(x1, training)

        # Scale the numerical variables
        x1_num = self.scale_X(x1, training)

        processed_x = (torch.from_numpy((pd.concat(objs=(x1_num, x1_cat), axis=1)).to_numpy())).float()

        if y is not None:
            y = self.scale_y(y)

        return processed_x, (y if isinstance(y, torch.Tensor) else None)

    @staticmethod
    def replace_nas(df):
        """Replaces NAs with medians or modes appropriately"""
        df2 = df.copy(deep=True)
        for col in df2.columns:
            # Replace numerical data with median
            if df2[col].dtype != object:
                df2[col].fillna((df2[col].median()), inplace=True)
            # Replace categorical data with mode
            else:
                df2[col].fillna((df2[col].mode()), inplace=True)
        return df2

    @staticmethod
    def has_no_infs(df):
        """Returns True if column has +ve or -ve infinity"""
        df_infs = df.isin([np.inf, -np.inf])
        return (df_infs.sum() == True).all()

    def replace_infs(self, df):
        if self.has_no_infs(df):
            return df
        else:
            df2 = df.copy(deep=True)
            # Replace with column max if inf and column min if -inf
            for col in df2.columns:
                m1 = df.loc[df[col] != np.inf, col].max()
                m2 = df.loc[df[col] != -np.inf, col].min()
                df2[col].replace(np.inf, m1, inplace=True)
                df2[col].replace(-np.inf, m2, inplace=True)
            return df2

    def encode(self, df, training):
        """Encodes categorical data using one hot encoder"""
        if training:
            # Initialise encoder
            self.encoder = OneHotEncoder()
            encoded_array = self.encoder.fit_transform(df[self.cat_var]).toarray()
        else:
            encoded_array = self.encoder.transform(df[self.cat_var]).toarray()

        feature_labels = self.encoder.categories_
        feature_labels = np.array(feature_labels).ravel()
        encoded_df = pd.DataFrame(data=encoded_array, columns=feature_labels)

        return encoded_df

    def scale_X(self, df, training):
        """Scales numerical variables of X using a standard scaler"""
        num_data = df[self.num_vars].copy(deep=True)
        if training:
            scaler = StandardScaler().fit(num_data)
            self.scaler = scaler

        scaled_num_data = pd.DataFrame(data=self.scaler.transform(num_data), columns=self.num_vars)

        return scaled_num_data

    def scale_y(self, y):
        """converts a pd dataframe to a tensor"""
        if not torch.is_tensor(y):
            target_data = y.copy(deep=True)
            scaled_y = target_data.to_numpy()
            return (torch.from_numpy(scaled_y)).float()
        else:
            return y

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        X, Y = self._preprocessor(x=x, y=y, training=True)
        self.error_fn = F.mse_loss
        data_loader = self.batch_loader(x=X, y=Y, shuffle=True)
        model = self.Model(input_size=self.input_size, activation=self.activation, layers=self.layers,
                           neurons=self.neurons, output_activation=self.output_activation)
        self.model = model
        self.optim = Adam(self.model.parameters(), lr=self.learning_rate)

        # We run over each epoch and train data in batches
        for epoch in range(self.nb_epoch):
            epoch_error = 0
            count = 0
            for x_t, y_t in data_loader:
                # reset gradients
                self.optim.zero_grad()

                # Forward Pass
                prediction = self.model.forward(x_t)

                # Loss
                loss = self.error_fn(prediction, y_t, reduction='mean')

                self.losses.append(loss.item())

                epoch_error += loss.item() * len(x_t)
                count += len(x_t)

                # Backward pass
                loss.backward()

                # Update parameters
                self.optim.step()

            self.errors.append(epoch_error / len(x))

            # Print a 10th of the way through epochs

            """if (epoch + 1) % (self.nb_epoch / 10) == 0:
                print(f'Epoch{epoch + 1}, Loss: {(epoch_error / count) ** 0.5}')"""

        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        X, _ = self._preprocessor(x, training=False)
        y_tensor = self.model.forward(X)
        return y_tensor.detach().numpy()

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        X, Y = self._preprocessor(x, y=y, training=False)
        pred = self.predict(x)

        return np.sqrt(mean_squared_error(pred, y))

    def batch_loader(self, x, y, shuffle=True):
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return loader

    def y_unscale(self, y):
        m, s = (self.y_scaler.mean_[0]), (self.y_scaler.scale_[0])
        return torch.mul(torch.add(y, m), s)


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in model.pickle.
    """
    with open('model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments: params(dict): Dictionary with a list for each key. Each key is the parameter we want to test  for the
    parameters we're looking to test

    Returns:
        best_params(dict); Dictionary with a list with the optimal parameter for each key

    """

    params = {'nb_epoch': [500, 750, 1000], 'lr': [1e-2, 1e-3, 1e-4], 'batch_size': [32, 64, 128, 256],
              'activation': ['relu'], 'layers': [2, 3, 4], 'neurons': [16, 32, 64],
              'output_activation': ['relu']}
    classifier = GridSearchCV(estimator=Regressor(x_train), cv=5, param_grid=params, verbose=1,
                              scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error')

    classifier.fit(x_train, y_train)
    print(classifier.best_params_)
    print(classifier.best_score_)
    print(classifier.best_estimator_)
    save_regressor(classifier.best_estimator_)
    return classifier.best_estimator_


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    X_train, X_test, Y_train, Y_test = train_test_split(
        x_train, y_train, test_size=0.10, random_state=101)

    # Training

    regressor = Regressor(X_train, nb_epoch=1000)
    regressor.fit(X_train, Y_train)
    save_regressor(regressor)

    # Error
    error = np.sqrt(regressor.score(X_test, Y_test))
    print("\nRegressor error: {}\n".format(error))

    RegressorHyperParameterSearch(x_train, y_train)
    pass


if __name__ == "__main__":
    example_main()

    """output_label = "median_house_value"

    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    X_train, X_test, Y_train, Y_test = train_test_split(
        x_train, y_train, test_size=0.10, random_state=101)

    # Training and outputting the best parameter on all the data

    regressor = Regressor(x_train, nb_epoch=500, lr=1e-3, batch_size=32, activation='relu', layers=4, neurons=64,
                          output_activation='relu')
    regressor.fit(x_train, y_train)

    save_regressor(regressor)"""
