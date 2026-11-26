"""Run the verified video preset or a detector configuration supplied by the caller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("video", type=Path)
    parser.add_argument("--preset", choices=("video", "accuracy"), default="video")
    args, extra = parser.parse_known_args()
    preset_name = "final_config.json" if args.preset == "video" else "accuracy_config.json"
    config = json.loads((ROOT / preset_name).read_text())
    command = [sys.executable, str(ROOT / "zone_monitor.py"),
               str(args.video.expanduser().resolve()), "--model", str(ROOT / config["model"])]
    for key in ("conf", "imgsz", "device", "tracker", "exit_confirm", "track_ttl"):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    return subprocess.call(command + extra, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
