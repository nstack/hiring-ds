#!/usr/bin/env python3

import argparse
import zipfile
import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics
import sklearn.ensemble


def read_df(file_name):
    print("Reading %s..." % file_name)
    df = pd.read_csv(file_name, dtype={'id': str, 'Customer ID': str,
                                       'Product SKU': str, 'Price': str,
                                       'price': float, 'profit': float})
    original_len = len(df)
    print('Read in %d rows...' % original_len)
    df.dropna(inplace=True)
    print("Dropped %d rows containing nan..." % (original_len - len(df)))

    if 'Customer ID' in df.columns:
        df.rename(columns={'Customer ID': 'id'}, inplace=True)
        print("Renamed Customer ID to id...")

    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'].str[:10], format="%d/%m/%Y")
    else:
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")
    print("Parsed dates...")

    if 'Order Date' in df.columns:
        df.rename(columns={'Order Date': 'date'}, inplace=True)
        print("Renamed Order Date to date...")

    if 'Price' in df.columns:
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
        df.rename(columns={'Price': 'price'}, inplace=True)
        print("Renamed Price to price...")

    original_len = len(df)
    df = df[df['id'] != '-1']
    print("Dropped %d rows containing -1 in id column..." % (original_len - len(df)))
    print()

    return df


def create_healthy_binary_target(df):
    average_days_between_orders = df.groupby('id')['date'].apply(
        lambda x: (x.max() - x.min()) / len(x))
    average_days_between_orders = average_days_between_orders[
        average_days_between_orders != datetime.timedelta(0)]

    print("Percentiles for average days between orders for returning customers:")
    print(average_days_between_orders.describe(percentiles=np.arange(0, 1, 0.1)))
    print()

    quantile = average_days_between_orders.quantile(q=0.9)
    print("90%% of the customers that make repeated purchase make it after %d days." %
          quantile.days)
    latest_date = df['date'].max()

    cut_off = latest_date - quantile
    print("Latest date found in the dataset: %s." % latest_date)
    print("Customers who have purchases after %s are considered healthy." % cut_off)

    df['healthy'] = df.groupby('id')['date'].transform(max) > cut_off
    print("Created %d transactions corresponding to healthy customers." % np.sum(df['healthy']))

    unique_customers = df.groupby('id')['healthy'].tail(1)
    num_unique = len(unique_customers)
    num_healthy = np.sum(unique_customers)
    print("Number of unique customers: %d." % num_unique)
    print("Number of healthy customers: %d." % num_healthy)
    print("Proportion of healthy customers: %0.2f." % (num_healthy / num_unique))

    return df


def add_features(df):
    print("Adding secondsSinceRegistration feature...")
    df['secondsSinceRegistration'] = df.groupby('id')['date'].transform(
        lambda x: (x - x.min())).apply(lambda x: x.total_seconds())
    print("Adding numOfTransactions features...")
    df['numOfTransactions'] = df.groupby('id')['date'].transform(lambda x: np.argsort(x) + 1)

    return df


def train_test_splitting(df):
    train_ids, _ = train_test_split(df['id'].unique())
    train_mask = df['id'].isin(train_ids)
    df_train = df[train_mask]
    df_test = df[~train_mask]
    print("Size of training data set: %d." % len(df_train))
    print("Size of test data set: %d." % len(df_test))
    train_mean = df_train['healthy'].mean()
    print("Proportion of transactions corresponding to healthy customers in training dataset: "
          "%0.2f." % train_mean)
    test_mean = df_test['healthy'].mean()
    print("Proportion of transactions corresponding to healthy customers in test dataset: %0.2f." %
          test_mean)

    return df_train, df_test


def features(df):
    features_ = ['price', 'secondsSinceRegistration', 'numOfTransactions']
    if 'profit' in df.columns:
        features_.append('profit')
    return features_


def create_find_roc_metric(df_test):
    orders_test_latest = df_test.sort_values('date').groupby('id').tail(1)

    def find_roc_metric(model):
        healthy_proba = model.predict_proba(orders_test_latest[features(df_test)])[:, 1]
        roc_metric = sklearn.metrics.roc_auc_score(orders_test_latest['healthy'], healthy_proba)
        return roc_metric

    return find_roc_metric


def find_best_model(metric_func, df_train):
    print("Finding the best random forest model...")
    curr_area = -100
    curr_model = None

    search_dict = {}

    for min_impurity_decrease in list(np.arange(0.0, 0.3, 0.02)):
        rf = sklearn.ensemble.RandomForestClassifier(min_impurity_decrease=min_impurity_decrease)
        rf.fit(df_train[features], df_train['healthy'])
        area = metric_func(rf)
        if area > curr_area:
            curr_area = area
            curr_model = rf

        search_dict[min_impurity_decrease] = area
        print("Min impurity: %0.3f. Area under ROC curve: %0.3f." % (min_impurity_decrease, area))

    return curr_model


def print_feature_importances(model, df):
    print("Most important features:")
    print(sorted(list(zip(features(df), model.feature_importances_)), key=lambda x: x[1],
                 reverse=True))


def customers_health(model, df):
    if 'health_score' in df:
        del df['health_score']

    print("Adding health score column...")
    orders_latest = df.sort_values('date').groupby('id').tail(1)
    proba = model.predict_proba(orders_latest[features(df)])[:, 1]
    health_score = pd.DataFrame({'id': orders_latest['id'], 'health_score': proba})
    return df.join(health_score.set_index('id'), how='inner', on='id')


def print_csv(df):
    print("Printig csv...")
    print()
    print("id,health_score")
    print(df.groupby('id')['health_score'].max().to_csv())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train health model.')
    parser.add_argument('file_name', type=str, help='an integer for the accumulator')

    args = parser.parse_args()
    file_name = args.file_name

    with zipfile.ZipFile("orders.zip", 'r') as zip_ref:
        print("Extracting orders.zip...")
        zip_ref.extractall('.')

    df = read_df(file_name)
    df = create_healthy_binary_target(df)
    df = add_features(df)
    df_train, df_test = train_test_splitting(df)
    roc_metric = create_find_roc_metric(df_test)
    model = find_best_model(roc_metric, df_train)
    print_feature_importances(model, df)
    df = customers_health(model, df)
    print("Saving full csv...")
    df.to_csv(file_name[:-4] + '-new.csv', index=False)
    print_csv(df)
