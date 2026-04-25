#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sqlite3
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GeoPackage vector layer to PMTiles using ogr2ogr."
    )
    parser.add_argument("--input-gpkg", required=True, help="Input GeoPackage path.")
    parser.add_argument("--output-pmtiles", required=True, help="Output PMTiles path.")
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional input layer name. Defaults to all layers supported by ogr2ogr.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output PMTiles if it already exists.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional PMTiles name metadata.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional PMTiles description metadata.",
    )
    parser.add_argument(
        "--minzoom",
        type=int,
        default=None,
        help="Optional PMTiles minimum zoom metadata.",
    )
    parser.add_argument(
        "--maxzoom",
        type=int,
        default=None,
        help="Optional PMTiles maximum zoom metadata.",
    )
    return parser.parse_args()


def get_ogr2ogr_candidates() -> list[str]:
    candidates: list[str] = []
    preferred_paths = [Path("/usr/bin/ogr2ogr"), Path("/usr/local/bin/ogr2ogr")]
    path_binary = shutil.which("ogr2ogr")

    for candidate in preferred_paths + ([Path(path_binary)] if path_binary else []):
        if not candidate:
            continue
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in candidates:
            candidates.append(candidate_str)

    if candidates:
        return candidates
    raise SystemExit("ogr2ogr is required but was not found on PATH.")


def infer_gpkg_layers(input_gpkg: Path) -> list[str]:
    with sqlite3.connect(input_gpkg) as connection:
        cursor = connection.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type IN ('features', 'attributes') ORDER BY table_name"
        )
        return [str(row[0]) for row in cursor.fetchall() if row and row[0]]


def build_command(args: argparse.Namespace, ogr2ogr_bin: str, layer_name: str | None) -> list[str]:
    input_gpkg = Path(args.input_gpkg).expanduser().resolve()
    output_pmtiles = Path(args.output_pmtiles).expanduser().resolve()

    if not input_gpkg.exists():
        raise SystemExit(f"Input GeoPackage not found: {input_gpkg}")

    if output_pmtiles.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_pmtiles}. Pass --overwrite to replace it.")

    output_pmtiles.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ogr2ogr_bin]
    if args.overwrite:
        cmd.append("-overwrite")

    cmd.extend(["-f", "PMTiles", str(output_pmtiles), str(input_gpkg)])

    if layer_name:
        cmd.append(str(layer_name))

    if args.name:
        cmd.extend(["-dsco", f"NAME={args.name}"])
    if args.description:
        cmd.extend(["-dsco", f"DESCRIPTION={args.description}"])
    if args.minzoom is not None:
        cmd.extend(["-dsco", f"MINZOOM={args.minzoom}"])
    if args.maxzoom is not None:
        cmd.extend(["-dsco", f"MAXZOOM={args.maxzoom}"])

    return cmd


def main() -> int:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg).expanduser().resolve()
    output_pmtiles = Path(args.output_pmtiles).expanduser().resolve()
    layer_name = args.layer

    if layer_name is None:
        inferred_layers = infer_gpkg_layers(input_gpkg)
        if len(inferred_layers) == 1:
            layer_name = inferred_layers[0]

    failures: list[str] = []

    for ogr2ogr_bin in get_ogr2ogr_candidates():
        cmd = build_command(args, ogr2ogr_bin, layer_name)
        if output_pmtiles.exists() and args.overwrite:
          output_pmtiles.unlink()

        try:
            subprocess.run(cmd, check=True)
            print(f"Used ogr2ogr: {ogr2ogr_bin}")
            if layer_name:
                print(f"Layer: {layer_name}")
            print(f"Wrote PMTiles: {output_pmtiles}")
            print(f"Size bytes: {output_pmtiles.stat().st_size}")
            return 0
        except subprocess.CalledProcessError as exc:
            if output_pmtiles.exists():
                output_pmtiles.unlink()
            signal_note = ""
            if exc.returncode < 0:
                signal_note = f" (terminated by signal {-exc.returncode})"
            failures.append(f"{ogr2ogr_bin}: exit code {exc.returncode}{signal_note}")

    raise SystemExit("PMTiles conversion failed with all available ogr2ogr binaries:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    raise SystemExit(main())