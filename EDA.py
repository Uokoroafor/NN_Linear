import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':
    housing_data = pd.read_csv('housing.csv')
    print(housing_data.describe())
    (housing_data.describe()).to_csv('housing_data_des.csv')
    print(housing_data.shape)
    print(housing_data.columns)
    print(housing_data.head())
    target_col = housing_data['median_house_value']
    ax = housing_data.plot.hist(figsize=(20, 15), bins=30)
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()
    housing_nas = housing_data.isna()


    def count_nas(df):
        df_na = (df.isna())
        na_sums = df_na.sum(axis=0)
        return na_sums


    na_counts = count_nas(housing_data)
    print(na_counts)


    # Dealing with NAs
    def replace_nas(df):
        df2 = df.copy(deep=True)
        for col in df2.columns:

            if df2[col].dtype != object:
                df2[col].fillna((df2[col].median()), inplace=True)
            else:
                df2[col].fillna((df2[col].mode()), inplace=True)
        return df2


    print(housing_data.dtypes)
    housing_new = replace_nas(housing_data)
    print(count_nas(housing_new))
    print(housing_data.dtypes)
    print(housing_new.dtypes)


    def has_no_infs(df):
        df_infs = df.isin([np.inf, -np.inf])
        return (df_infs.sum() == True).all()


    def replace_infs(df):
        if has_no_infs(df):
            return df
        else:
            df2 = df.copy(deep=True)
            # Replace with column max if inf and column min if -inf
            for col in df2.columns:
                m1 = df.loc[df[col] != np.inf, col].max()
                m2 = df.loc[df[col] != np.inf, col].min()
                df2[col].replace(np.inf, m1, inplace=True)
                df2[col].replace(-np.inf, m2, inplace=True)
            return df2


    inf_check = has_no_infs(housing_data)

    housing2 = replace_infs(housing_data)

    categories = housing2['ocean_proximity'].unique()
    print(categories)

    ohe = OneHotEncoder()
    ohe.fit(housing2[['ocean_proximity']])
    prox_array = ohe.fit_transform(housing2[['ocean_proximity']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    print(feature_labels)
    prox_df = pd.DataFrame(data=prox_array, columns=feature_labels)
    print(prox_df.head())
    ohe.fit
    ###########

    ###########

    cat_vars = ['ocean_proximity']
    target_var = ['median_house_value']
    num_vars = [k for k in housing_data.columns.tolist() if (k not in cat_vars) and (k not in target_var)]

    num_data = housing_data[num_vars].copy(deep=True)
    scaler = StandardScaler().fit(num_data)
    print(scaler.mean_)
    print(scaler.scale_)
    scaled_num_data = pd.DataFrame(data=scaler.transform(num_data), columns=num_vars)
    mean_dict = {k: i for k, i in zip(num_vars, scaler.mean_)}
    mean_sd = {k: i for k, i in zip(num_vars, scaler.scale_)}
    print(scaled_num_data.describe())

    target_data = housing_data[target_var].copy(deep=True)
    y_scaler = StandardScaler().fit(target_data)
    scaled_y = pd.DataFrame(data=y_scaler.transform(target_data), columns=target_var)

    processed_X = pd.concat(objs=(scaled_num_data, prox_df), axis=1)

    y_array = scaled_y.to_numpy()
    X_array = processed_X.to_numpy()
    input_size, output_size = X_array.shape[1], 1

    y_tensor, X_tensor = torch.from_numpy(y_array), torch.from_numpy(X_array)

    model = nn.Linear(input_size, 1)

    print(model.weight)
    print(model.bias)
    print(list(model.parameters()))

    # Loss Function
    score = F.mse_loss

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)

    count = 0
    for x_t, y_t in loader:
        print(count + 1)
        count += 1

    pass
