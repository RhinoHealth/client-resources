#!/bin/bash

if [ $# -ne 3 ]; then
  >&2 echo "Usage: $0 input_dir output_dir csv_file"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
CSV_FILE=$3

DIR=$(dirname "$(readlink -f "$BASH_SOURCE")")

set -x
set -e

python $DIR/dataprep_gc.py --input_csv ${CSV_FILE} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR}
