"""One-time conversion of the MANO files so they load without chumpy.
Run once from the project root: python convert_mano.py"""
import os
import pickle
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"  # the models/ folder next to this script


class ChumpyStandIn:
    """Takes the place of chumpy's classes while the files are read; keeps their stored data."""

    def __setstate__(self, state):
        self.state = state


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return type(name, (ChumpyStandIn,), {})  # a stand-in that keeps chumpy's class name
        return super().find_class(module, name)


def to_array(obj):
    """Turns chumpy's stored data back into a NumPy array."""
    if type(obj).__name__ == "Ch":  # a plain chumpy array: the numbers are in "x"
        return np.asarray(obj.state["x"])
    if type(obj).__name__ == "Select":  # a selection of elements from another chumpy array
        source = to_array(obj.state["a"])
        return source.ravel()[obj.state["idxs"]].reshape(obj.state["preferred_shape"])
    raise ValueError(f"unknown chumpy class {type(obj).__name__}")


for side in ["RIGHT", "LEFT"]:
    path = MODELS / f"MANO_{side}.pkl"
    original = MODELS / f"MANO_{side}_original.pkl"
    if os.path.exists(original):
        print(f"{side}: already converted")
        continue
    with open(path, "rb") as f:
        data = Unpickler(f, encoding="latin1").load()
    for key, value in data.items():
        if isinstance(value, ChumpyStandIn):
            data[key] = to_array(value)
    os.rename(path, original)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"{side}: converted")