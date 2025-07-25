from dataclasses import dataclass
from typing import Any
import numpy as np


class Node:
    def __init__(self, probabilities: list[(Any, float)]):
        total = sum([weighting for token, weighting in probabilities])
        self.probabilities = [(token, weighting / total) for token, weighting in probabilities]

    def generate_token(self):
        value = np.random.random()
        chosen_token = self.probabilities[0][0]

        cum_prob = 0
        for token, prob in self.probabilities:
            cum_prob += prob
            if cum_prob >= value:
                chosen_token = token
                break

        return chosen_token


@dataclass()
class Edge:
    start: Node
    token: Any
    end: Node


class PDFA:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.state: Node = None
        self.initial_state = None

    def add_node(self, probabilities: list[(Any, float)], initial_state=False) -> Node:
        node = Node(probabilities)
        self.nodes.append(node)
        if initial_state:
            self.initial_state = node
        return node

    def add_edge(self, node1: Node, token: Any, node2: Node) -> Edge:
        edge = Edge(node1, token, node2)
        self.edges.append(edge)
        return edge

    def reset(self):
        self.state = self.initial_state

    def update(self, token: Any):
        for edge in self.edges:
            if edge.start == self.state and edge.token == token:
                self.state = edge.end
                return
        raise ValueError(f"Token {token} was not valid for current state.")

    def generate_string(self):
        self.reset()
        while True:
            token = self.state.generate_token()
            yield token
            self.update(token)
