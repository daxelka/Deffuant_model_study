from deffuant_model import DeffuantModel
import distribution_tools as tools
import numpy as np

# Set up model parameters
N_nodes: int = 100
epsilon = 0.2
mu = 0.5
n_steps = int(1e04)

# Initiate the model
model = DeffuantModel(N_nodes, epsilon, mu)

# generate distribution for initial conditions
initial_opinion = tools.uniform_opinion(N_nodes)
# Set initial condition
model.set_opinion(initial_opinion)

# Run the model for n_steps
model.run(n_steps)
final_opinion = model.get_opinion()

# Show opinion distribution
model.show_opinion_distribution(final_opinion)
