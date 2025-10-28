#!/bin/bash
set -euo pipefail
python3 -m baseline_cnn_mnist.train --epochs 10 --batch-size 128 --lr 1e-3 --num-workers 4 "$@"
