import json

from PIL import Image
import torch

from few_step_diffusion.config import load_config
from few_step_diffusion import sd_data


def test_coco_order_prompt_hash_and_caption_only_vsd(tmp_path, monkeypatch):
    config = load_config("configs/sd15.yaml")
    config["data"].update(root=str(tmp_path), train_limit=2, batch_size=2, download=False)
    (tmp_path/"annotations").mkdir()
    (tmp_path/"train2017").mkdir()
    content = {"images": [{"id": 9, "file_name": "9.jpg"}, {"id": 2, "file_name": "2.jpg"}],
               "annotations": [{"id": 17, "image_id": 2, "caption": "wrong later caption"},
                               {"id": 10, "image_id": 9, "caption": "a car"},
                               {"id": 4, "image_id": 2, "caption": "a red door on the left"}]}
    (tmp_path/"annotations/captions_train2017.json").write_text(json.dumps(content))
    image = Image.new("RGB", (512, 512), "blue")
    image.paste("red", (0, 0, 256, 512))
    for name in ("2.jpg", "9.jpg"):
        image.save(tmp_path/"train2017"/name)
    rows = sd_data.caption_records(config)
    assert [r["image_id"] for r in rows] == [2, 9]
    assert rows[0]["caption_id"] == 4
    stream = sd_data.CaptionStream(config)
    x, prompts = stream.batch(5, torch.device("cpu"))
    torch.manual_seed(1234)
    resumed_x, resumed_prompts = stream.batch(5, torch.device("cpu"))
    torch.testing.assert_close(x, resumed_x, rtol=0, atol=0)
    assert prompts == resumed_prompts
    assert x[:, 0, :, :100].mean() > .9  # Captions stay paired with unflipped images.
    original_hash = sd_data.manifest_hash(rows)
    rows[0]["caption"] = "changed caption"
    assert sd_data.manifest_hash(rows) != original_hash

    def forbidden(*args):
        raise AssertionError("Caption-only VSD may not read or download training images")
    monkeypatch.setattr(sd_data, "image_path", forbidden)
    vsd = sd_data.CaptionStream(config, images=False)
    images, prompts = vsd.batch(5, torch.device("cpu"))
    assert images is None and len(prompts) == 2
