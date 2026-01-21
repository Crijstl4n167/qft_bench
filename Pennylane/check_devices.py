import pennylane as qml

for name in ["default.qubit", "lightning.qubit", "lightning.gpu"]:
    try:
        dev = qml.device(name, wires=2)
        print(f"OK: {name} -> {type(dev)}")
    except Exception as e:
        print(f"FAIL: {name} -> {e}")

