import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataExtract:
    @staticmethod
    def get_url(year: int, month: int) -> str:
        return f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month}.parquet"

    @staticmethod
    def get_data_response(url: str):
        return requests.get(url)

    def download_data(self, year: int, month: int):
        url = self.get_url(year, month)
        resp = self.get_data_response(url)
        file_name = f"taxi_data_{year}_{month}.parquet"
        with open(file_name, "wb") as f:
            f.write(resp.content)
        return file_name

    @staticmethod
    def load_data(file_name: str):
        df = pd.read_parquet(file_name)
        return df


class FeatureEngineering:
    @staticmethod
    def get_trip_duration(df: pd.DataFrame):
        df["trip_duration"] = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
        df["trip_duration"] = df["trip_duration"].dt.total_seconds() / 60
        return df

    @staticmethod
    def get_pickup_hour(df: pd.DataFrame):
        df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
        return df

    @staticmethod
    def get_day_of_week(df: pd.DataFrame):
        df["day_of_week"] = df["lpep_pickup_datetime"].dt.day_name()
        return df

    def add_new_features(self, df: pd.DataFrame):
        df = self.get_trip_duration(df)
        df = self.get_pickup_hour(df)
        df = self.get_day_of_week(df)
        return df

    @staticmethod
    def group_by_hour(df: pd.DataFrame):
        df_group_hour = df.groupby("pickup_hour").mean()
        df_group_hour = df_group_hour.reset_index()
        return df_group_hour

    @staticmethod
    def label_encode(data):
        le = LabelEncoder()
        return le.fit_transform(data)

    def get_x_y(self, df_feat):
        y_data = df_feat["tip_amount"]
        X_data = df_feat.drop("tip_amount", axis=1)
        X_data["store_and_fwd_flag"] = self.label_encode(df_feat["store_and_fwd_flag"])
        X_data["day_of_week"] = self.label_encode(df_feat["day_of_week"])
        return X_data, y_data

    @staticmethod
    def split_data(X_data, y_data, val_size=0.15, test_size=0.15):
        X_train, X_test_val, y_train, y_test_val = train_test_split(
            X_data, y_data, test_size=val_size + test_size, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test_val, y_test_val, test_size=0.5, random_state=42
        )

        data = {}
        data["X_train"] = X_train
        data["y_train"] = y_train
        data["X_val"] = X_val
        data["y_val"] = y_val
        data["X_test"] = X_test
        data["y_test"] = y_test
        return data

    @staticmethod
    def stand_to_df(data, scaler, data_name):
        trans_data = scaler.transform(data.get(data_name))
        df_trans_data = pd.DataFrame(trans_data, columns=data.get(data_name).columns)
        return df_trans_data

    def standardize_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data.get("X_train"))
        data["X_train"] = self.stand_to_df(data, scaler, "X_train")
        data["X_val"] = self.stand_to_df(data, scaler, "X_val")
        data["X_test"] = self.stand_to_df(data, scaler, "X_test")
        return data

    @staticmethod
    def remove_outliers(
        X_data: pd.DataFrame,
        y_data: pd.DataFrame,
        outlier_field: str,
        outlier_threshold=3,
    ):
        new_X_data = []
        new_y_data = []
        out_col_idx = X_data.columns.tolist().index(outlier_field)
        y_data_values = y_data.values
        for row_idx, row in enumerate(X_data.values):
            if row[out_col_idx] >= outlier_threshold:
                continue
            new_X_data.append(row)
            new_y_data.append(y_data_values[row_idx])
        new_X_data = pd.DataFrame(new_X_data, columns=X_data.columns)
        new_y_data = pd.Series(new_y_data)
        return new_X_data, new_y_data


class FeatureSelection:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def lasso_feature_selection(self):
        alphas = [0, 0.1, 0.01, 0.001]
        fig, axes = plt.subplots(len(alphas), figsize=[8, 20])
        for idx, alpha in enumerate(alphas):
            clf = linear_model.Lasso(alpha=alpha)
            clf.fit(self.X_data, self.y_data)
            sns.barplot(
                y=self.X_data.columns,
                x=clf.coef_,
                label=f"alpha: {alpha}",
                ax=axes[idx],
            )

    def random_forest_feature_selection(self):
        reg = RandomForestRegressor()
        reg.fit(self.X_data, self.y_data)
        sns.barplot(y=self.X_data.columns, x=reg.feature_importances_)


class Models:
    @staticmethod
    def get_linear_reg_results(data):
        results = {}
        reg = linear_model.LinearRegression()
        reg.fit(data.get("X_train"), data.get("y_train"))
        results["train"] = mean_squared_error(
            reg.predict(data.get("X_train")), data.get("y_train")
        )
        results["val"] = mean_squared_error(
            reg.predict(data.get("X_val")), data.get("y_val")
        )
        results["test"] = mean_squared_error(
            reg.predict(data.get("X_test")), data.get("y_test")
        )
        return results

    @staticmethod
    def get_random_forest_reg_results(data):
        results = {}
        reg = RandomForestRegressor(100)
        reg.fit(data.get("X_train"), data.get("y_train"))
        results["train"] = mean_squared_error(
            reg.predict(data.get("X_train")), data.get("y_train")
        )
        results["val"] = mean_squared_error(
            reg.predict(data.get("X_val")), data.get("y_val")
        )
        results["test"] = mean_squared_error(
            reg.predict(data.get("X_test")), data.get("y_test")
        )
        return results

    @staticmethod
    def get_isolation_forest_preds(data):
        iso = IsolationForest()
        return iso.fit_predict(data)


class TaxiDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data.values.astype(np.float32))
        self.y_data = torch.from_numpy(y_data.values.astype(np.float32))

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


def get_test_results(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    all_preds = []
    all_ys = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.float().unsqueeze(1)
            X = X.to(device)
            pred = model(X)
            all_preds += pred.tolist()
            all_ys += y.tolist()
    test_loss = mean_squared_error(all_preds, all_ys)
    return test_loss


def train_model(
    train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, device
):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    model = model.to(device)
    res_history = defaultdict(list)
    train_dataloader_copy = train_dataloader
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_dataloader):
            y = y.float().unsqueeze(1)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                val_loss = get_test_results(val_dataloader, model, loss_fn, device)
                train_loss = get_test_results(
                    train_dataloader_copy, model, loss_fn, device
                )
                print(f"Epoch: {epoch}, Batch: {batch}/{num_batches}")
                print(f"Train loss: {round(train_loss, 4)}")
                print(f"Val loss: {round(val_loss, 4)}")
                res_history["val"].append(val_loss)
                res_history["train"].append(train_loss)
    return train_loss, val_loss, res_history


def get_neural_network_results(
    data, epochs=10, batch_size=128, learning_rate=0.001, plot_results=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    train_data = TaxiDataset(data.get("X_train"), data.get("y_train"))
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_data = TaxiDataset(data.get("X_val"), data.get("y_val"))
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_data = TaxiDataset(data.get("X_test"), data.get("y_test"))
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    input_dim = len(set(data.get("X_train")))
    print("Using input dimension: ", input_dim)
    loss_fn = nn.MSELoss()
    model = NeuralNetwork(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, val_loss, res_history = train_model(
        train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, device
    )
    test_loss = get_test_results(test_dataloader, model, loss_fn, device)
    train_loss = get_test_results(train_dataloader, model, loss_fn, device)
    results = {"train": train_loss, "test": test_loss, "val": val_loss}
    if plot_results:
        plt.plot(
            range(len(res_history.get("train"))),
            res_history.get("train"),
            label="Train",
        )
        plt.plot(
            range(len(res_history.get("val"))), res_history.get("val"), label="Val"
        )
        plt.legend()
    return results
