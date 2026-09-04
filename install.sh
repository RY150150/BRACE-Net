#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
(
  cd brace_net/cpp/pointnet2_batch
  python setup.py install
)
