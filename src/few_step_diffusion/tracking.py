"""Noninteractive W&B setup with local offline logging when credentials are absent."""

import json
import netrc
import os
from pathlib import Path
from urllib.parse import urlparse

from tqdm.auto import tqdm

from .runtime import atomic_json
from .config import dataset_name


def has_credentials():
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        auth = netrc.netrc(os.environ.get("NETRC")).authenticators(host)
        return bool(auth and auth[2])
    except (OSError, netrc.NetrcParseError):
        return False


def resolve_mode(config):
    requested = os.environ.get("WANDB_MODE", config.get("wandb", {}).get("mode", "auto"))
    if requested not in ("auto", "online", "offline", "disabled"):
        raise ValueError("WANDB_MODE must be auto, online, offline, or disabled")
    return ("online" if has_credentials() else "offline") if requested == "auto" else requested


class TrainingTracker:
    def __init__(self, config, environment, output, resume=False):
        self.config, self.environment, self.output = config, environment, Path(output)
        self.resume = resume
        self.run = None
        self.wandb = None
        self.failed_logging = False

    def __enter__(self):
        mode = resolve_mode(self.config)
        if mode == "disabled":
            return self
        if mode == "online" and not has_credentials():
            raise ValueError("Online W&B requires `wandb login` or WANDB_API_KEY; use offline or auto")
        import wandb

        self.wandb = wandb
        settings = self.config.get("wandb", {})
        metadata_file = self.output / "wandb_run.json"
        previous = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        # Online runs can resume. W&B offline sessions cannot resume history;
        # create a separate session and keep its parent ID for later inspection.
        run_id = previous.get("id") if self.resume and mode == "online" else None
        kwargs = dict(
            project=os.environ.get("WANDB_PROJECT", settings.get("project", "few-step-diffusion")),
            entity=os.environ.get("WANDB_ENTITY", settings.get("entity")),
            group=os.environ.get("WANDB_RUN_GROUP", settings.get("group")) or
            ({"COCO-2017": "sd15", "CelebA-HQ": "celebahq"}.get(dataset_name(self.config), "cifar10")),
            name=os.environ.get("WANDB_NAME", settings.get("name")) or
            f"{self.config['name']}-seed{self.config['seed']}",
            tags=settings.get("tags", []), job_type=self.config["train"]["mode"],
            config={**self.config, "environment": self.environment},
            dir=str(self.output.resolve()), mode=mode, id=run_id,
            resume="allow" if run_id else None, save_code=False,
            settings=wandb.Settings(init_timeout=15, console="off", disable_git=True),
        )
        try:
            self.run = wandb.init(**kwargs)
        except Exception:
            requested = os.environ.get("WANDB_MODE", settings.get("mode", "auto"))
            if requested != "auto" or mode != "online":
                raise
            tqdm.write("W&B online initialization failed; recording this session offline.")
            wandb.teardown()
            kwargs.update(mode="offline", id=None, resume=None)
            self.run = wandb.init(**kwargs)
            mode = "offline"
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")
        self.run.summary["status"] = "running"
        atomic_json(metadata_file, {"id": self.run.id, "mode": mode,
                    "project": self.run.project, "parent_id": previous.get("id") if self.resume else None,
                    "directory": self.run.dir})
        tqdm.write(f"W&B tracking: {mode}, run={self.run.id}")
        return self

    def log(self, values):
        if self.run is not None and not self.failed_logging:
            try:
                self.run.log(values)
            except Exception:
                self.failed_logging = True
                tqdm.write("W&B logging failed; training and local JSON logs continue.")

    def image(self, path, step, stage):
        if self.run is not None and self.config.get("wandb", {}).get("log_images", True):
            self.log({"global_step": step, f"samples/{stage}_steps": self.wandb.Image(str(path))})

    def summarize(self, values):
        if self.run is not None:
            try:
                self.run.summary.update(values)
            except Exception:
                tqdm.write("W&B summary unavailable; run.json retains the training summary.")

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            self.summarize({"status": "complete" if exc_type is None else "interrupted"
                            if issubclass(exc_type, (KeyboardInterrupt, InterruptedError)) else "failed"})
            try:
                self.run.finish(exit_code=0 if exc_type is None else 1)
            except Exception:
                tqdm.write("W&B could not finish cleanly; local run files are retained.")
        return False
