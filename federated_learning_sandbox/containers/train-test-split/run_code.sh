#!/bin/bash

DIR=$(dirname "$(readlink -f "$BASH_SOURCE")")

set -x
set -e

python $DIR/train_test_split_gc.py
