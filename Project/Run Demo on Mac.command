#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  if ! command -v python3.13 >/dev/null; then
    print 'Install Python 3.13 for macOS from python.org, then reopen this launcher.'
    exit 1
  fi
  python3.13 -m venv .venv
fi
if ! .venv/bin/python -c 'import ultralytics, cv2, lap' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements-lock.txt
fi
.venv/bin/python run_project.py CAVIAR_Walk2_Reentry.mp4 --save-video --out demonstration
