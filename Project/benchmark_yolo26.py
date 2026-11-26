"""Compare the COCO-pretrained YOLO26 medium detector at its native 640-pixel setting."""
import sys
from pathlib import Path
import benchmark_detectors

if __name__ == '__main__':
    benchmark_detectors.CANDIDATES = [('yolo26m_640', 'yolo26m.pt', 640)]
    if not sys.argv[1:]:
        sys.argv.extend(['--out', str(Path(__file__).resolve().parent / 'yolo26_benchmark')])
    benchmark_detectors.main()
