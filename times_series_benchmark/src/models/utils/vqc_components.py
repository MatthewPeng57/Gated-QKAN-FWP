# Copyright 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# VQC components for batch processing with MetaCore.
# Be careful when both inputs and weights (parameters) are batched.
#
# The layer helpers below (H_layer / RX_layer / RY_layer / RZ_layer /
# entangling_layer) follow the standard PennyLane variational-classifier
# tutorial layout for building a hardware-efficient ansatz; see
# https://pennylane.ai/qml/demos/tutorial_variational_classifier

import torch

import pennylane as qml

from functools import partial


### VQC

def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates.
	"""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)

def RX_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx in range(w.shape[-1]):
		qml.RX(w[..., idx], wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx in range(w.shape[-1]):
		qml.RY(w[..., idx], wires=idx)

def RZ_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx in range(w.shape[-1]):
		qml.RZ(w[..., idx], wires=idx)


def entangling_layer(nqubits):
	"""Layer of CNOTs followed by another shifted layer of CNOT.
	"""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])
	

def cycle_entangling_layer(nqubits):
	for i in range(0, nqubits):  
		qml.CNOT(wires=[i, (i + 1) % nqubits])
		
##############



### VQC function

def quantum_net(inputs, q_weights, n_outputs):
	# n_dep = q_weights.shape[0] # 2025 06 23: Should we edit here???
	# n_qub = q_weights.shape[1]  # 2025 06 23: Should we edit here???
	n_dep = q_weights.shape[1] # 2025 06 23: Should we edit here???
	n_qub = q_weights.shape[2]  # 2025 06 23: Should we edit here???

	H_layer(n_qub)

	RY_layer(inputs)

	for k in range(n_dep):
		# print("k: {}".format(k)) # 2025 06 23: looks the intrepretation here is not correct!! (running through the batch_size???)
		entangling_layer(n_qub)
		RY_layer(q_weights[:,k,:]) # 2025 06 23: Should we edit here???


	return [qml.expval(qml.PauliZ(position)) for position in range(n_outputs)]


##############

class HideSignature:
	def __init__(self, partial_func):
		self.partial_func = partial_func

	def __call__(self, inputs, q_weights):
		return self.partial_func(inputs, q_weights)

##############

### VQC batch wrapper

class BatchVQC:
	def __init__(self, q_func):
		self.q_func = q_func

	def __call__(self, inputs, q_weights):

		# res_all = []
		# for input_item, q_weight_item in zip(inputs, q_weights):
		# 	res = self.q_func(input_item, q_weight_item) 
		# 	res_all.append(torch.stack(res))
		# print("inputs.shape: {}".format(inputs.shape))
		# print("q_weights.shape: {}".format(q_weights.shape))
		res_all = torch.stack(self.q_func(inputs, q_weights)).T
		# print("res_all: {}".format(res_all))
		# print("res_all.shape: {}".format(res_all.shape))

		return res_all


##############

def main():
	torch.manual_seed(0)

	B = 4
	n_qubits = 3
	depth = 2
	n_outputs = 2

	dev = qml.device("default.qubit", wires=n_qubits)

	vqc = BatchVQC(qml.QNode(HideSignature(partial(quantum_net, n_outputs = n_outputs)), dev, interface = "torch"))

	# inputs: no grad
	x = torch.randn(B, n_qubits, requires_grad=False)

	# weights: with grad
	q_weights = torch.randn(B, depth, n_qubits, requires_grad=True)

	# forward
	y = vqc(x, q_weights)          # (B, n_outputs)
	loss = (y**2).mean()           # scalar
	loss.backward()

	print("y.shape:", y.shape)
	print("loss:", float(loss))
	print("x.requires_grad:", x.requires_grad, "x.grad is None:", (x.grad is None))
	print("q_weights.requires_grad:", q_weights.requires_grad)
	print("q_weights.grad is None:", (q_weights.grad is None))
	if q_weights.grad is not None:
		print("q_weights.grad shape:", q_weights.grad.shape)
		print("grad finite:", torch.isfinite(q_weights.grad).all().item())
		print("grad abs max:", q_weights.grad.abs().max().item())

	# -------------------------
	# Finite-difference sanity check (one element)
	# -------------------------
	eps = 1e-4
	b, k, j = 0, 0, 0  # check grad for q_weights[b,k,j]
	with torch.no_grad():
		w_plus = q_weights.clone()
		w_minus = q_weights.clone()
		w_plus[b, k, j] += eps
		w_minus[b, k, j] -= eps

		y_plus = vqc(x, w_plus)
		y_minus = vqc(x, w_minus)
		loss_plus = (y_plus**2).mean()
		loss_minus = (y_minus**2).mean()
		fd = (loss_plus - loss_minus) / (2 * eps)

	autograd_val = q_weights.grad[b, k, j].detach()
	print("\n[FD check] index (b,k,j) =", (b, k, j))
	print("finite-diff:", float(fd))
	print("autograd   :", float(autograd_val))
	print("abs error  :", float((fd - autograd_val).abs()))










	return






if __name__ == '__main__':
	main()
