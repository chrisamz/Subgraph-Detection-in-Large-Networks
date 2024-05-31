# data_preprocessing.py

"""
Data Preprocessing Module for Subgraph Detection in Large Networks

This module contains functions for collecting, cleaning, normalizing, and preparing
network data for subgraph detection algorithms.

Techniques Used:
- Data cleaning
- Normalization
- Graph construction
- Handling missing data

Libraries/Tools:
- pandas
- numpy
- NetworkX
"""

import pandas as pd
import numpy as np
import networkx as nx
import os

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        pass

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        data = data.dropna()
        return data

    def normalize_data(self, data, columns):
        """
        Normalize specified columns in the data.
        
        :param data: DataFrame, input data
        :param columns: list, column names to normalize
        :return: DataFrame, normalized data
        """
        data[columns] = (data[columns] - data[columns].min()) / (data[columns].max() - data[columns].min())
        return data

    def construct_graph(self, data, source_col, target_col, weight_col=None):
        """
        Construct a graph from the data.
        
        :param data: DataFrame, input data
        :param source_col: str, column name for source nodes
        :param target_col: str, column name for target nodes
        :param weight_col: str, column name for edge weights (optional)
        :return: NetworkX graph
        """
        if weight_col:
            graph = nx.from_pandas_edgelist(data, source=source_col, target=target_col, edge_attr=weight_col)
        else:
            graph = nx.from_pandas_edgelist(data, source=source_col, target=target_col)
        return graph

    def save_graph(self, graph, filepath):
        """
        Save the graph to a file.
        
        :param graph: NetworkX graph
        :param filepath: str, path to save the graph
        """
        nx.write_gpickle(graph, filepath)
        print(f"Graph saved to {filepath}")

    def preprocess(self, raw_data_filepath, processed_data_dir, source_col, target_col, weight_col=None, normalize_cols=[]):
        """
        Execute the full preprocessing pipeline.
        
        :param raw_data_filepath: str, path to the input data file
        :param processed_data_dir: str, directory to save processed data
        :param source_col: str, column name for source nodes
        :param target_col: str, column name for target nodes
        :param weight_col: str, column name for edge weights (optional)
        :param normalize_cols: list, column names to normalize
        :return: NetworkX graph, preprocessed graph
        """
        # Load data
        data = self.load_data(raw_data_filepath)

        # Clean data
        data = self.clean_data(data)

        # Normalize data
        if normalize_cols:
            data = self.normalize_data(data, normalize_cols)

        # Construct graph
        graph = self.construct_graph(data, source_col, target_col, weight_col)

        # Save graph
        os.makedirs(processed_data_dir, exist_ok=True)
        graph_filepath = os.path.join(processed_data_dir, 'preprocessed_graph.gpickle')
        self.save_graph(graph, graph_filepath)

        return graph

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/network_data.csv'
    processed_data_dir = 'data/processed/'
    source_col = 'source'  # Example source node column
    target_col = 'target'  # Example target node column
    weight_col = 'weight'  # Example weight column (optional)
    normalize_cols = ['weight']  # Example columns to normalize (optional)

    preprocessing = DataPreprocessing()

    # Preprocess the data
    graph = preprocessing.preprocess(raw_data_filepath, processed_data_dir, source_col, target_col, weight_col, normalize_cols)
    print("Data preprocessing completed and graph saved.")
