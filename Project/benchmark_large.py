"""Compare the COCO-pretrained large detector at its native 640-pixel setting."""
import sys
from pathlib import Path
import benchmark_detectors

if __name__ == '__main__':
    benchmark_detectors.CANDIDATES = [('yolo11l_640', 'yolo11l.pt', 640)]
    if not sys.argv[1:]:
        sys.argv.extend(['--out', str(Path(__file__).resolve().parent / 'large_benchmark')])
    benchmark_detectors.main()
