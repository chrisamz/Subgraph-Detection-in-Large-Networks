# query_based_algorithms.py

"""
Query-Based Algorithms for Subgraph Detection in Large Networks

This module contains functions for detecting subgraphs in large networks using query-based methods.

Techniques Used:
- Query optimization
- Indexing
- Search algorithms

Libraries/Tools:
- NetworkX
- pandas
- numpy

"""

import networkx as nx
import numpy as np
import pandas as pd
import os

class QueryBasedSubgraphDetection:
    def __init__(self, graph):
        """
        Initialize the QueryBasedSubgraphDetection class with a NetworkX graph.
        
        :param graph: NetworkX graph
        """
        self.graph = graph

    def node_attribute_query(self, attribute_name, attribute_value):
        """
        Detect subgraphs based on node attributes.
        
        :param attribute_name: str, name of the node attribute
        :param attribute_value: value of the node attribute to query
        :return: NetworkX graph, subgraph containing nodes with the specified attribute value
        """
        query_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get(attribute_name) == attribute_value]
        subgraph = self.graph.subgraph(query_nodes).copy()
        return subgraph

    def edge_attribute_query(self, attribute_name, attribute_value):
        """
        Detect subgraphs based on edge attributes.
        
        :param attribute_name: str, name of the edge attribute
        :param attribute_value: value of the edge attribute to query
        :return: NetworkX graph, subgraph containing edges with the specified attribute value
        """
        query_edges = [(u, v) for u, v, attr in self.graph.edges(data=True) if attr.get(attribute_name) == attribute_value]
        subgraph = self.graph.edge_subgraph(query_edges).copy()
        return subgraph

    def shortest_path_query(self, source, target):
        """
        Detect subgraphs based on the shortest path between two nodes.
        
        :param source: node, source node
        :param target: node, target node
        :return: NetworkX graph, subgraph containing the shortest path between the source and target nodes
        """
        path = nx.shortest_path(self.graph, source=source, target=target)
        subgraph = self.graph.subgraph(path).copy()
        return subgraph

    def custom_query(self, query_function):
        """
        Detect subgraphs based on a custom query function.
        
        :param query_function: function, custom query function that returns a list of nodes or edges
        :return: NetworkX graph, subgraph containing nodes or edges returned by the custom query function
        """
        query_result = query_function(self.graph)
        if isinstance(query_result[0], tuple):
            subgraph = self.graph.edge_subgraph(query_result).copy()
        else:
            subgraph = self.graph.subgraph(query_result).copy()
        return subgraph

    def save_subgraph(self, subgraph, filepath):
        """
        Save the detected subgraph to a file.
        
        :param subgraph: NetworkX graph, detected subgraph
        :param filepath: str, path to save the subgraph
        """
        nx.write_gpickle(subgraph, filepath)
        print(f"Subgraph saved to {filepath}")

if __name__ == "__main__":
    processed_data_dir = 'data/processed/'
    graph_filepath = os.path.join(processed_data_dir, 'preprocessed_graph.gpickle')
    query_output_dir = os.path.join(processed_data_dir, 'query_subgraphs/')
    os.makedirs(query_output_dir, exist_ok=True)

    # Load the preprocessed graph
    graph = nx.read_gpickle(graph_filepath)

    query_detection = QueryBasedSubgraphDetection(graph)

    # Detect subgraphs based on node attributes
    node_attribute_subgraph = query_detection.node_attribute_query('attribute_name', 'attribute_value')
    node_attribute_subgraph_filepath = os.path.join(query_output_dir, 'node_attribute_subgraph.gpickle')
    query_detection.save_subgraph(node_attribute_subgraph, node_attribute_subgraph_filepath)

    # Detect subgraphs based on edge attributes
    edge_attribute_subgraph = query_detection.edge_attribute_query('attribute_name', 'attribute_value')
    edge_attribute_subgraph_filepath = os.path.join(query_output_dir, 'edge_attribute_subgraph.gpickle')
    query_detection.save_subgraph(edge_attribute_subgraph, edge_attribute_subgraph_filepath)

    # Detect subgraphs based on the shortest path between two nodes
    shortest_path_subgraph = query_detection.shortest_path_query('source_node', 'target_node')
    shortest_path_subgraph_filepath = os.path.join(query_output_dir, 'shortest_path_subgraph.gpickle')
    query_detection.save_subgraph(shortest_path_subgraph, shortest_path_subgraph_filepath)

    # Detect subgraphs based on a custom query function
    def custom_query_function(graph):
        # Example custom query: find all nodes with degree greater than 5
        return [n for n in graph.nodes if graph.degree[n] > 5]

    custom_query_subgraph = query_detection.custom_query(custom_query_function)
    custom_query_subgraph_filepath = os.path.join(query_output_dir, 'custom_query_subgraph.gpickle')
    query_detection.save_subgraph(custom_query_subgraph, custom_query_subgraph_filepath)

    print("Query-based subgraph detection completed and subgraphs saved.")
