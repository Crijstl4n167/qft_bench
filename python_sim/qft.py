import numpy as np
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
      result = np.kron(result, state)
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

def QFT(state: np.ndarray) -> np.ndarray:
  dim = int(np.log2(len(state)))
  for target in range(dim):
    state = apply(create(dim, [target], [H]), state)
    for control in range(target+1, dim):
      state = apply(
        CG(dim, control, target, R(control-target+1)),
        state
      )
  return state # NOTE: swapped bit order now

def SQFT(state: np.ndarray) -> np.ndarray:
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

def IQFT(state: np.ndarray) -> np.ndarray:
  raise NotImplemented

def INVQFT(state: np.ndarray) -> np.ndarray:
  dim = int(np.log2(len(state)))
  for target in reversed(range(dim)):
    for control in reversed(range(target+1, dim)):
      state = apply(
        CG(dim, control, target, R(control-target+1).conj().T),
        state
      )
    state = apply(create(dim, [target], [H]), state)
  return state # NOTE: swapped bit order now

#############
### Adder ###
#############
def qadd(a: int, b: int) -> int:
  if a < 0 or b < 0:
    raise ValueError('qadd_using_qft only supports non-negative integers')

  n = max(a.bit_length(), b.bit_length()) + 1
  a_bits = list(f'{a:0{n}b}')

  state = kron([
    qs0 if bit == '0' else qs1
    for bit in a_bits
  ])

  state = QFT(state)

  phase_layer = kron([
    np.array([
      [1, 0],
      [0, np.exp(2j * np.pi * b / (2 ** (n - i)))]
    ], dtype=complex)
    for i in range(n)
  ])

  state = apply(phase_layer, state)
  state = INVQFT(state)

  probs = np.abs(state).flatten() ** 2
  return int(np.argmax(probs))

# Optimized adder
def qadd_optimized(a: int, b: int) -> int:
  if a < 0 or b < 0:
    raise ValueError('qadd only supports non-negative integers')

  n = max(a.bit_length(), b.bit_length()) + 1
  a_bits = list(f'{a:0{n}b}')
  b_bits = list(f'{b:0{n}b}')

  state = kron([
    qs0 if bit == '0' else qs1
    for bit in (a_bits + b_bits)
  ])

  dim = 2 * n
  a_idx = list(range(n))
  b_idx = list(range(n, 2 * n))

  phase_cache = {k: np.exp(2j * np.pi / (2 ** k)) for k in range(1, n + 1)}

  def apply_single_qubit(state: np.ndarray, gate: np.ndarray, target: int) -> np.ndarray:
    reshaped = state.reshape([2] * dim)
    moved = np.moveaxis(reshaped, target, 0)
    updated = gate @ moved.reshape(2, -1)
    restored = np.moveaxis(updated.reshape([2] + [2] * (dim - 1)), 0, target)
    return restored.reshape(-1, 1)

  def apply_controlled_phase(state: np.ndarray, control: int, target: int, phase: complex) -> np.ndarray:
    reshaped = state.reshape([2] * dim)
    idx = [slice(None)] * dim
    idx[control] = 1
    idx[target] = 1
    reshaped[tuple(idx)] *= phase
    return reshaped.reshape(-1, 1)

  def qft_on_register(state: np.ndarray, reg: List[int]) -> np.ndarray:
    for i, target in enumerate(reg):
      state = apply_single_qubit(state, H, target)
      for j in range(i + 1, len(reg)):
        control = reg[j]
        state = apply_controlled_phase(state, control, target, phase_cache[j - i + 1])
    return state

  def inv_qft_on_register(state: np.ndarray, reg: List[int]) -> np.ndarray:
    for i in reversed(range(len(reg))):
      target = reg[i]
      for j in reversed(range(i + 1, len(reg))):
        control = reg[j]
        state = apply_controlled_phase(state, control, target, np.conjugate(phase_cache[j - i + 1]))
      state = apply_single_qubit(state, H, target)
    return state

  state = qft_on_register(state, a_idx)

  for j, target in enumerate(a_idx):
    for k, control in enumerate(b_idx):
      p = n - j - (n - 1 - k)
      if p >= 1:
        state = apply_controlled_phase(state, control, target, phase_cache[p])

  state = inv_qft_on_register(state, a_idx)

  probs = np.abs(state).reshape(2 ** n, 2 ** n) ** 2
  a_marginal = probs.sum(axis=1)
  return int(np.argmax(a_marginal))

if __name__ == '__main__':
  print(qadd(100,200))

# vim:ts=2 sw=2 et:
