from pathlib import Path
import runpy
import subprocess
import sys

import pytest


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux process identity check")
def test_handoff_binds_adoption_to_process_start_time_and_arguments():
    module = runpy.run_path("scripts/launch_scale50k.py")
    record_process, matches = module["process_record"], module["process_matches"]
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        record = record_process(child.pid)
        assert matches(record)
        assert not matches({**record, "start_ticks": "0"})
        assert not matches({**record, "command": ["unexpected-other-process"]})
        assert not matches({**record, "cwd": "/other/workspace"})
    finally:
        child.terminate()
        child.wait(timeout=5)
    assert not matches(record)
