#!/bin/bash
set -eux -o pipefail

# Decrypt all files with a .enc extension.
find . -type f -name '*.enc' -exec bash -c 'python ./app/custom/decrypt_code.py "$1" "${1%.enc}"' bash {} ';'

exec "$@"
