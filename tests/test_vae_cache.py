from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

import few_step_diffusion.vae_cache as cache
from few_step_diffusion.recent_performance import RealEncoder, grouped_microbatch


def prepared_cache(tmp_path, monkeypatch):
    records = [{"image_id": 10}, {"image_id": 20}, {"image_id": 30}]
    protocol = {"shape": [3, 8, 2, 2], "count": 3, "model_revision": "synthetic-test-model"}
    monkeypatch.setattr(cache, "cache_protocol", lambda config: (protocol, records))
    values = np.zeros((3, 8, 2, 2), dtype=np.float32)
    for i in range(3):
        values[i, :4] = i
        values[i, 4:] = i+1
    np.save(tmp_path/"posterior.npy", values)
    np.save(tmp_path/"image_ids.npy", np.array([10, 20, 30]))
    (tmp_path/"metadata.json").write_text(json.dumps({"status": "complete", "protocol": protocol}))
    return protocol, records


def test_cached_statistics_preserve_image_order_and_fresh_rng(tmp_path, monkeypatch):
    _, records = prepared_cache(tmp_path, monkeypatch)
    reader = cache.PosteriorCache(tmp_path, {})
    posterior = reader.batch([records[2], records[0]])
    assert posterior.mean[:, 0, 0, 0].tolist() == [2, 0]
    assert posterior.std[:, 0, 0, 0].tolist() == [3, 1]
    encoder = RealEncoder.__new__(RealEncoder)
    encoder.main = torch.device("cpu")
    encoder.enabled = True
    encoder.bundle = SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=.18215)))
    assert encoder.start(posterior) is posterior
    torch.manual_seed(72)
    expected = (posterior.mean+posterior.std*torch.randn_like(posterior.mean))*.18215
    rng = torch.get_rng_state()
    torch.manual_seed(72)
    actual = encoder.finish(posterior, encoder.start(posterior))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.equal(actual, encoder.finish(posterior, encoder.start(posterior)))


def test_incomplete_or_foreign_cache_is_rejected(tmp_path, monkeypatch):
    protocol, _ = prepared_cache(tmp_path, monkeypatch)
    for metadata in [{"status": "building", "protocol": protocol},
                     {"status": "complete", "protocol": {**protocol, "model_revision": "wrong-model"}}]:
        (tmp_path/"metadata.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="incomplete or belongs"):
            cache.PosteriorCache(tmp_path, {})


def test_cached_grouping_keeps_same_effective_batch_and_caption_order():
    def loader(stream, index, device, real=False):
        values = torch.full((2, 4, 2, 2), float(index))
        return cache.PosteriorBatch(values, torch.ones_like(values)), [str(index)]*2
    posterior, captions = grouped_microbatch(loader, None, 4, "cpu", True, 4)
    assert posterior.mean.shape[0] == 8
    assert posterior.mean[:, 0, 0, 0].tolist() == [4, 4, 5, 5, 6, 6, 7, 7]
    assert captions == ["4", "4", "5", "5", "6", "6", "7", "7"]


def test_cached_training_needs_no_helper_gpu_allocation(monkeypatch):
    monkeypatch.setenv("FSD_RECENT_VAE_CACHE", "/synthetic/test/cache")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: pytest.fail("No helper allocation should be queried"))
    class VAE:
        config = SimpleNamespace(scaling_factor=.5)
        def to(self, device):
            assert device == torch.device("cpu")
            return self
        def decode(self, values):
            return SimpleNamespace(sample=values*2)
    bundle = SimpleNamespace(device=torch.device("cpu"), unet=torch.nn.Linear(1, 1), vae=VAE())
    encoder = RealEncoder(bundle, {"vae_helper"})
    assert encoder.device == torch.device("cpu")
    torch.testing.assert_close(bundle.decode(torch.ones(2, 4, 2, 2)), torch.full((2, 4, 2, 2), 4.))
