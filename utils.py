
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, Imputer, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(file_dir, date_col, id_col, price_col):
    df = pd.read_csv(file_dir, date_parser=True, usecols=[date_col, id_col, price_col])
    
    df[price_col] = df[price_col].astype(str)
    df[price_col] = df[price_col].str.replace(',','')
    
    df.replace([-1, "null","nan","NaN",'NaT', 'nat'], np.nan, inplace = True)
    
    df[price_col] = pd.to_numeric(df[price_col])
    negative_values = df[price_col] <= 0.0
    df.loc[negative_values, price_col] = np.nan
    
    mean_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    df[price_col] = mean_imputer.fit_transform(np.array(df[price_col]).reshape(-1,1))
    
    df.dropna(subset=[id_col, date_col], inplace=True)
    df[date_col] = pd.to_datetime(df[date_col])
        
    return df


def pivot_df_feature(df, date_col, id_col, price_col):
    df_pivot_1 = pd.pivot_table(df, values=price_col, index=[id_col], columns=df[date_col].dt.month , aggfunc=np.sum)
    df_pivot_1 = df_pivot_1.add_suffix('_price')
    
    df_pivot_2 = pd.pivot_table(df, values=price_col, index=[id_col], columns=df[date_col].dt.month , aggfunc=np.count_nonzero)
    df_pivot_2 = df_pivot_2.add_suffix('_count')
    
    df_pivot = pd.concat([df_pivot_1,df_pivot_2], axis=1, join_axes=[df_pivot_1.index])
    df_pivot.fillna(0, inplace=True)
    
    return df_pivot

def pivot_df_target(df, date_col, id_col, price_col):
    df_pivot = pd.pivot_table(df, values=price_col, index=[id_col], columns=df[date_col].dt.month , aggfunc=np.sum)
    df_pivot = df_pivot.add_suffix('_price')
    
    df_pivot.fillna(0, inplace=True)
    
    return df_pivot

def split_data(df, date_col, id_col, price_col):
    df = df.sort_values(by=date_col)
    #getting the last month as our target data
    target_data = df[df[date_col].dt.strftime("%m-%y")==df[date_col].iloc[-1].strftime("%m-%y")]
    feature_data = df[df[date_col].dt.strftime("%m-%y")!=df[date_col].iloc[-1].strftime("%m-%y")]
    
    #pivoting target and feature data
    target_data = pivot_df_target(target_data, date_col, id_col, price_col)
    feature_data = pivot_df_feature(feature_data, date_col, id_col, price_col)
    
    #get mutual customers and split data into train and test
    idx = feature_data.index.intersection(target_data.index)
    target_data = (target_data.loc[idx]).sort_index()
    feature_data = (feature_data.loc[idx]).sort_index()
    
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.30)
    
    return X_train, X_test, y_train, y_test

def min_max_scaler(x):
    float_array = x.values.astype(float)
    min_max_scaler = MinMaxScaler()
    scaled = pd.DataFrame(min_max_scaler.fit_transform(float_array.reshape(-1,1)))
    
    return scaled

