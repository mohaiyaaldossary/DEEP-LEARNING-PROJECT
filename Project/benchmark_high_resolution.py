"""Run the same CPU protocol at 960 pixels on the existing medium checkpoint."""
import sys
from pathlib import Path
import benchmark_detectors

if __name__ == '__main__':
    benchmark_detectors.CANDIDATES = [('yolo11m_960', 'yolo11m.pt', 960)]
    if not sys.argv[1:]:
        sys.argv.extend(['--out', str(Path(__file__).resolve().parent / 'high_resolution_benchmark')])
    benchmark_detectors.main()
