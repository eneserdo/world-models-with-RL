import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import PathResolutionError, resolve_ctrl_file  # noqa: E402


class CtrlPathTests(unittest.TestCase):
    def test_missing_flag_is_named(self):
        for bad in (None, ""):
            with self.assertRaises(PathResolutionError) as ctx:
                resolve_ctrl_file(bad)
            self.assertIn("--logdir", str(ctx.exception))

    def test_missing_checkpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PathResolutionError) as ctx:
                resolve_ctrl_file(tmp)
            self.assertIn("best.tar", str(ctx.exception))

    def test_valid_path_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "ctrl").mkdir()
            Path(tmp, "ctrl", "best.tar").write_bytes(b"x")
            self.assertEqual(os.path.join(tmp, "ctrl", "best.tar"), resolve_ctrl_file(tmp))


if __name__ == "__main__":
    unittest.main()
