# Architecture

- `data/` - dataset generation, loading and the simulated environment wrappers.
- `envs/` - the CarRacing environment adapters.
- `models/` - VAE, MDN-RNN and controller definitions.
- `utils/` - shared helpers (`misc.py`, `paths.py`).
- `traincontroller.py` / `test_controller.py` - the evaluation entry points.
