"""Exercise temporary-pool disposal and restoration under an active TMS scope."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch_npu
from torch_memory_saver import configure_subprocess, torch_memory_saver as tms


def run(mode):
    if mode == "preload" and "torch_memory_saver" not in os.environ.get("LD_PRELOAD", ""):
        with configure_subprocess():
            subprocess.run([sys.executable, "-u", __file__, mode], check=True)
        return
    torch.npu.set_device(0)
    tms.hook_mode = mode
    for cycle in range(3):
        with tms.region(tag="disable-test", enable_cpu_backup=True):
            kept = torch.arange(4096, device="npu", dtype=torch.float32)
            for i in range(8):
                with tms.disable():
                    temporary = torch.ones(4096, device="npu")
                    assert temporary.sum().item() == 4096
            try:
                with tms.disable():
                    temporary = torch.full((128,), 3, device="npu")
                    raise ValueError("scope cleanup")
            except ValueError as exc:
                assert str(exc) == "scope cleanup"
            after = torch.full((128,), 7, device="npu")
            assert after.sum().item() == 896
        address = kept.data_ptr()
        tms.pause("disable-test")
        tms.resume("disable-test")
        assert kept.data_ptr() == address
        assert torch.equal(kept.cpu(), torch.arange(4096, dtype=torch.float32))
        del kept, after, temporary
        torch.npu.empty_cache()
    native = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "torch_memory_saver_hook" in line})
    assert native and all("site-packages" in item for item in native)
    print("PASS", json.dumps({"mode": mode, "cycles": 3, "disable_scopes": 27,
                              "exception_restoration": True, "backup_exact": True, "native": native}), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
