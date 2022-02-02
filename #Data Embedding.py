#Extracts data from CSV file
#Creates paramterized feature map and ansatz
#Utilizes quantum neural network for training and SPSA for optimization

from cProfile import label
from turtle import color
import numpy as np
import math
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit, assemble
from qiskit.circuit import ParameterVector, Parameter
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap, TwoLocal
#from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.neural_networks import CircuitQNN, OpflowQNN

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sympy import rotations


data_file = open("SomeData.csv")                    #Reads data from csv file
data_array = np.loadtxt(data_file, delimiter=",")   #Creates array from sample data
sample_size, nb_features = data_array.shape         #Gathers sample size and feature size values from data array


#Need some sort of a classification in place, so we can "classifiy". Update this code to create classification for our data.
labels = [0, 1]
y = (data_array[:,1] > 0.5).astype(int)     #Assigns label 1 or 0, depending on data value being larger than 0.5 or not.
#print(y)

x_train, x_test, y_train, y_test = train_test_split(data_array, y, test_size=0.2)   #Splits data_array 80:20 between train and test
print(x_test)
print(y_test)

#Preps the backend and the quantum instance
sim = Aer.get_backend('aer_simulator')
shots = 8092
qinst = QuantumInstance(sim, shots)


#Puts together the feature map
map_z = ZZFeatureMap(feature_dimension=nb_features, reps=2, entanglement='linear')
map_z.assign_parameters({k:v for (k,v) in zip(map_z.parameters, x_train[0])}).decompose().draw('mpl', scale=0.7)

#Puts together the ansatz
def get_two_locals(nb_features, rotations, var_form_rep, ent):
    return TwoLocal(num_qubits=nb_features, rotation_blocks=rotations, entanglement_blocks='cx', entanglement=ent, reps=var_form_rep)

ansatz= get_two_locals(nb_features, ['ry', 'rz'], 2, 'linear')
#ansatz.decompose().draw('mpl', scale=0.7)
weights = np.random.random(len(ansatz.parameters))  #Assigns weights, it oftens initiated as random numbers
ansatz.assign_parameters({k:v for (k,v) in zip(ansatz.parameters, weights)}).decompose().draw('mpl', scale=0.7)

#Quantum Neural Network training happens here
var_circuit = map_z.compose(ansatz)
var_circuit.draw('mpl')
#plt.show()

def parity(x, num_classes):
    return f"{x:b}".count("1") % num_classes

def one_qubit_binary(x):
    return x % 2

num_classes = 2
qnn = CircuitQNN(circuit = var_circuit,
                input_params=map_z.parameters,
                weight_params=ansatz.parameters,
                interpret=one_qubit_binary,
                output_shape=num_classes,
                gradient=None,
                quantum_instance=qinst)

output = qnn.forward(x_train[0], weights)   #Forward pass
#print(output.shape)

_, update = qnn.backward(x_train[0], weights)   #Backward pass

#Training Section


