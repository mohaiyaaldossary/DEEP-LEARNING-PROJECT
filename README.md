# Person Detection and Tracking for Restricted Area Monitoring

ARTI 502 Deep Learning · Group 2 · Imam Abdulrahman Bin Faisal University

This prototype detects people in prerecorded video with YOLO11m and follows them with ByteTrack. Draw a polygon on the floor of the scene to define a restricted area. When a tracked person enters, the monitor displays a warning and saves the video timestamp, track ID, and a snapshot. A confirmed exit allows a later re-entry to generate a new alert.

## Run the demo

To watch the recorded demonstration, open [Demo.mp4](Demo.mp4). To run the monitor yourself, install Python 3.13 and use the launcher for your operating system. First-time setup requires internet access for Python packages.

- **Windows** — Double-click `Run Demo on Windows.bat` in the repository root.
- **macOS** — Open Terminal in the repository root and run `cd Project`, followed by `zsh "Run Demo on Mac.command"`.

The launchers process the included Walk2 clip with a saved polygon. Press **Space** to pause or **Q** to stop. Each run writes an annotated video, event and tracking CSV files, and alert snapshots into its own folder under `Project/demonstration/`. For headless execution, polygon editing, tests, and benchmark commands, see [START HERE.txt](START%20HERE.txt).

## Evaluation

The study compared ten detector configurations on a person-detection subset of COCO with 1,920 training, 240 validation, and 240 test images. YOLO11m at 640 pixels achieved **82.68% test F1**, with 88.49% precision and 77.59% recall.

On the separate Walk3 video, the monitor matched all three annotated zone entries and produced zero unmatched alerts. Across the full scene it recovered 39.75% of annotated person boxes. Both video clips came from the same camera, so these results describe that viewpoint.

## Project materials

| Material | Contents |
| --- | --- |
| [Report PDF](Final%20Report.pdf) | PDF edition of the study |
| [Project](Project/) | Application code, model weights, dataset splits, videos, saved runs, and benchmark results |
| [Verification](Verification/) | Recorded unit test results |
| [References](References.bib) | Bibliography |

The CAVIAR footage is attributed to the EC Funded CAVIAR project, IST 2001 37540. See the [Walk2 source notes](Project/CAVIAR_Walk2_Reentry_source.txt) and [Walk3 source notes](Project/CAVIAR_Walk3_Test_source.txt) for attribution, transformations, and license details.
