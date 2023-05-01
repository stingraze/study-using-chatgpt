#Coded with ChatGPT (GPT-4) by Tsubasa Kato (gave prompt)
#Intended for experimentary use and personal study.
#2023/5/1
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Define the graph
n = 5  # Number of nodes
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]

G = nx.Graph()
G.add_nodes_from(np.arange(0, n, 1))
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

# Create the Max-Cut QUBO
qubo = QuadraticProgram()
_ = [qubo.binary_var(f'x{i}') for i in range(n)]
linear = {f'x{i}': 0 for i in range(n)}
quadratic = {f'x{i}x{j}': 1 if (i, j) in edges or (j, i) in edges else 0 for i in range(n) for j in range(i+1, n)}

qubo.minimize(linear=linear, quadratic=quadratic)

# Run the QAOA algorithm
backend = Aer.get_backend('statevector_simulator')
algorithm_globals.random_seed = 42
quantum_instance = QuantumInstance(backend, seed_simulator=42, seed_transpiler=42)
qaoa = QAOA(quantum_instance=quantum_instance, reps=3)

optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qubo)

# Print the results
x = result.x
print(f'Solution: {x}')
print(f'Cut size: {result.fval}')

# Plot the results
colors = ['r' if x[i] == 0 else 'b' for i in range(n)]
nx.draw(G, pos, node_color=colors, with_labels=True)
plt.show()
