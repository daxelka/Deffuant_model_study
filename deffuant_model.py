import matplotlib.pyplot as plt
import numpy as np
import random


class DeffuantModel:
    def __init__(self, N_nodes, confidence, cautiousness):
        self.N_nodes = N_nodes
        self.node_ids = range(self.N_nodes)
        self.opinions = []
        # Deffuant parameters
        self.confidence = confidence  # restricted to interval (0, 0.5]
        self.cautiousness = cautiousness  # convergence parameter, mu,  restricted to interval (0, 0.5]
        self.PRECISION = 0.01  # difference between opinions that are considered identical

    def interaction(self):
        # choose two nodes for interaction at random
        node1, node2 = random.sample(self.node_ids, 2)
        # get nodes' opinions
        value1 = self.opinions[node1]
        value2 = self.opinions[node2]
        # calculate the distance in nodes' opinion
        diff = abs(value1 - value2)
        # calculate opinion update
        if self.confidence > diff > self.PRECISION:
            self.opinions[node1] = value1 + self.cautiousness * (value2 - value1)
            self.opinions[node2] = value2 + self.cautiousness * (value1 - value2)

    def run(self, n_steps):
        for step in range(n_steps):
            self.interaction()


    def set_opinion(self, opinion_array):
        self.opinions = list(opinion_array)

    def get_opinion(self):
        return self.opinions

    def show_opinion_distribution(self, opinions):
        bins = [0.01 * n for n in range(100)]
        plt.hist(opinions, bins=bins, density=True)
        plt.title("Histogram of opinions")
        plt.show()


