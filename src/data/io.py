import pandas as pd


def load_data(path, selected_columns=None, cut=None):
    dataframe = pd.read_csv(path, low_memory=False)

    if selected_columns is None:
        selected_columns = ["SMILES", "conc"]

    dataframe = dataframe[list(selected_columns)]

    if cut is not None:
        dataframe = dataframe.iloc[:cut]

    return dataframe


def load_base_dataframe(config):
    return load_data(config["path"], config["selected_columns"], config.get("cut"))
