import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from qiskit import Aer
from qiskit.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import QuantumInstance
import networkx as nx

# Function to calculate the Max-Cut value given the graph and the partitions
def maxcut_value(graph, solution):
    cut_value = 0
    for (u, v, weight) in graph.edges.data("weight"):
        if solution[u] != solution[v]:
            cut_value += weight
    return cut_value

# Generate synthetic data (50 random graphs with 5 nodes)
num_graphs = 50
maxcut_solutions = []
maxcut_values = []
threshold = 4

# Create a QuantumInstance using the Qiskit Aer simulator
algorithm_globals.random_seed = 12345
quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator_statevector'), seed_simulator=algorithm_globals.random_seed, seed_transpiler=algorithm_globals.random_seed)

# Set up the QAOA algorithm
qaoa = QAOA(quantum_instance=quantum_instance, initial_point=[0.25, 0.25])

# Use the MinimumEigenOptimizer with QAOA to solve the Max-Cut problem
optimizer = MinimumEigenOptimizer(qaoa)

for _ in range(num_graphs):
    graph = nx.gnp_random_graph(5, 0.5)
    
    # Assign weights to the edges of the graph
    for u, v in graph.edges():
        graph[u][v]['weight'] = 1.0
    
    maxcut = Maxcut(graph)
    qp = maxcut.to_quadratic_program()
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)

    result = optimizer.solve(qubo)
    
    maxcut_solution = result.x
    cut_value = maxcut_value(graph, result.x)
    
    maxcut_solutions.append(maxcut_solution)
    maxcut_values.append(cut_value)
    print (str(cut_value) + "\n")

# Convert the list of solutions to a NumPy array
X = np.array(maxcut_solutions)

# Assign labels to each solution based on whether the Max-Cut value is above the threshold
y = np.array([1 if value > threshold else 0 for value in maxcut_values])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Test the perceptron
accuracy = perceptron.score(X_test, y_test)
print(f"Perceptron accuracy: {accuracy}")
