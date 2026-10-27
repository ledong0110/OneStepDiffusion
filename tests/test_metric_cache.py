import numpy as np
import torch

from few_step_diffusion.evaluation import metric_cache_context


def test_locally_cached_fid_statistics_load_with_weights_only(tmp_path):
    path = tmp_path / "statistics.pt"
    stats = {"mu": np.ones(3, dtype=np.float32), "sigma": np.eye(3, dtype=np.float64)}
    torch.save(stats, path)
    with metric_cache_context():
        loaded = torch.load(path, weights_only=True)
    np.testing.assert_array_equal(loaded["mu"], stats["mu"])
    np.testing.assert_array_equal(loaded["sigma"], stats["sigma"])
