"""Deterministic private-only archive of models, predictions and replay ledgers."""
import argparse
import json
from pathlib import Path
import zipfile
from tools.verify_mnq_oracle import verify
from tools.mnq_oracle_prepare import sha256


def package(directory, destination):
    directory, destination = Path(directory), Path(destination)
    verify(directory)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(directory.iterdir()):
            if file.is_file():
                entry = zipfile.ZipInfo("run01/"+file.name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, file.read_bytes())
    return {"file": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size, "visibility": "private_only_contains_derived_market_prices_and_models"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.directory, args.destination), indent=2))
