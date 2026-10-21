# Tests

`test_paths.py` covers the shared path resolution (no GPU needed). The data and environment tests need
the generated rollouts; a missing dataset is an error rather than a skip.
