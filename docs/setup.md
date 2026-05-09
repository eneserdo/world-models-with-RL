# Setup

1. `pip install -r requirements.txt`
2. Put the CarRacing rollouts under `data/` and the trained models under `models/`.
3. `python test_controller.py --logdir models/<run>` loads `ctrl/best.tar` and rolls out once on CPU.

Training writes checkpoints into the run's `ctrl/` directory; `examine_data.py` renders one batch for
inspection.
