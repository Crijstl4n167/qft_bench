import numpy as np
import scipy.sparse as sp
from typing import List

##############
### States ###
##############

qs0 = np.array([
    [1],
    [0]
], dtype=complex)

qs1 = np.array([
    [0],
    [1]
], dtype=complex)

#############
### Gates ###
#############

I = np.array([
  [1, 0],
  [0, 1]
], dtype=complex)

X = np.array([
  [0, 1],
  [1, 0]
], dtype=complex)

H = np.array([
  [1, 1],
  [1, -1]
], dtype=complex) * 1/np.sqrt(2)

R = lambda k: np.array([
  [1, 0],
  [0, np.exp(2j * np.pi / (2 ** k))]
], dtype=complex)

M00 = qs0 * qs0.conj().T
M11 = qs1 * qs1.conj().T
MLIST = [M00, M11]

#############
### Utils ###
#############

def kron(states: List[np.ndarray]) -> np.ndarray:
    result = states[0]
    for state in states[1:]:
      result = sp.kron(result, state)
    return result

def apply(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
  return gate @ state

def create(
    dim: int, 
    bits: List[int], 
    gates: List[np.ndarray]
) -> np.ndarray:
    base = [I for _ in range(dim)]
    for bit, gate in zip(bits, gates):
      base[bit] = apply(gate, base[bit])
    return kron(base)

########################
### Controlled Gates ###
########################

def CG(
    dim: int,
    control: int = 0, 
    target: int = 1, 
    gate: np.ndarray = X
) -> np.ndarray:
    i0 = [I for _ in range(dim)]
    i1 = [I for _ in range(dim)]
    i0[control] = M00
    i1[control] = M11
    i1[target] = gate
    return kron(i0) + kron(i1)

def SCG(
    dim: int,
    control: int = 0, 
    targets: List[int] = [1], 
    gates: List[np.ndarray] = [X]
) -> np.ndarray:
    i0 = [I for _ in range(dim)]
    i1 = [I for _ in range(dim)]
    i0[control] = M00
    i1[control] = M11
    for i, target in enumerate(targets):
      i1[target] = gates[i]
    return kron(i0) + kron(i1)

####################
### QFT Variants ###
####################

def QFTS(state: np.ndarray) -> np.ndarray:
  dim = int(np.log2(len(state)))
  for target in range(dim):
    state = apply(create(dim, [target], [H]), state)
    for control in range(target+1, dim):
      state = apply(
        CG(dim, control, target, R(control-target+1)),
        state
      )
  return state # NOTE: swapped bit order now

def SQFTS(state: np.ndarray) -> np.ndarray:
  dim = int(np.log2(len(state)))
  Rs = [R(i) for i in range(2, dim+1)]
  for bit in range(dim):
    state = apply(create(dim, [bit], [H]), state)
    if bit + 1 < dim:
      state = apply(
        SCG(
          dim,
          control=bit+1,
          targets=list(range(bit+1)),
          gates=Rs[:bit+1][::-1]
        ),
        state
      )
  return state # NOTE: swapped bit order now

def IQFTS(state: np.ndarray) -> np.ndarray:
  raise NotImplemented

# vim:ts=2 sw=2 et:
