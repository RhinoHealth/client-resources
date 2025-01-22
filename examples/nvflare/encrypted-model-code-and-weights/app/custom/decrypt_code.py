#!/usr/bin/env python
import argparse
import json
import logging
import sys
from pathlib import Path

from cryptography.fernet import Fernet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decrypt a file using a key from the GC run parameters")

    parser.add_argument("input_filename", help="input filename")
    parser.add_argument("output_filename", help="output filename")

    args = parser.parse_args()
    print("STARTING DECRYPTION")

    if not Path(args.input_filename).exists():
        print(f"File not found: {args.input_filename}", file=sys.stderr)
        sys.exit(1)

    if Path("/server-credentials").exists():
        secret_run_params_file_path = Path("/server-credentials/secret_run_params.json")
    elif Path("/input").exists():
        secret_run_params_file_path = Path("/input/secret_run_params.json")
    else:
        print("Could not find directory for secret_run_params.json!", file=sys.stderr)
        sys.exit(1)

    if not secret_run_params_file_path.is_file():
        print("secret_run_params.json file is missing.", file=sys.stderr)
        sys.exit(1)

    with secret_run_params_file_path.open("rb") as secret_run_params_file:
        try:
            secret_run_params = json.load(secret_run_params_file)
        except json.JSONDecodeError as exc:
            print(f"Error decoding JSON in secret_run_params.json: {str(exc)}", file=sys.stderr)
            sys.exit(1)

    try:
        key = secret_run_params["key"]
    except KeyError:
        print('Missing key in secret_run_params.json: "key"', file=sys.stderr)
        sys.exit(1)

    logging.info(f"Decrypting input file '{args.input_filename}'")

    encrypted = Path(args.input_filename).read_bytes()
    logging.info(f"Read {len(encrypted)} bytes of encrypted content")

    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted)
    logging.info(f"Decrypted contents resulted in {len(decrypted)} bytes")

    logging.info(f"Writing decrypted contents to output file '{args.output_filename}'")
    Path(args.output_filename).write_bytes(decrypted)

    logging.info("Done.")