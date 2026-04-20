#!/usr/bin/env python3

from __future__ import annotations

import argparse
import configparser
import getpass
import importlib
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWER_ROOT = REPO_ROOT / "phase_3_models" / "unet_site_models" / "wombat_mappingAI_viewer"
DEFAULT_HTML = REPO_ROOT / "cnn_mappingAI_viewer.html"
DEFAULT_PREFIX = "fvc_composition-viewer"
DEFAULT_CACHE_CONTROL = "public, max-age=31536000, immutable"
JSON_CACHE_CONTROL = "public, max-age=86400"
NO_CACHE_CONTROL = "no-cache"
DEFAULT_VIEWER_ROOT_RELATIVE = DEFAULT_VIEWER_ROOT.relative_to(REPO_ROOT)
PUBLISH_EXTENSIONS = {
    ".html",
    ".json",
    ".geojson",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}
TILE_SUFFIX_PATTERN = re.compile(r"tile(?P<tile_id>\d+)$")
SOURCE_TILE_PATTERN = re.compile(r"tiles_multispectral[._](?P<tile_id>\d+)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class PublishFile:
    source: Path
    relative_path: Path
    cache_control: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the publishable FVC composition viewer bundle to Cloudflare R2."
    )
    parser.add_argument("--bucket", default=os.environ.get("R2_BUCKET"), help="R2 bucket name.")
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="AWS shared-credentials profile to use for R2 access keys.",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("FVC_COMPOSITION_R2_PREFIX", DEFAULT_PREFIX),
        help="Object prefix inside the bucket.",
    )
    parser.add_argument(
        "--viewer-root",
        default=str(DEFAULT_VIEWER_ROOT),
        help="Local viewer data root containing dataset folders.",
    )
    parser.add_argument(
        "--html",
        default=str(DEFAULT_HTML),
        help="Path to cnn_mappingAI_viewer.html.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset folder to upload. Repeat for multiple datasets. Defaults to all datasets under the viewer root.",
    )
    parser.add_argument(
        "--include-root-index",
        action="store_true",
        help="Upload repo-root tiles_index.json if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be uploaded without contacting R2.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if sampled tiles are missing predictor COG, STAC, or thumbnail assets.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise SystemExit(f"Missing required environment variable: {name}")


def prompt_value(prompt: str, *, secret: bool = False) -> str:
    reader = getpass.getpass if secret else input
    value = reader(prompt).strip()
    if value:
        return value
    raise SystemExit(f"Missing required value for: {prompt.rstrip(': ')}")


def resolve_runtime_value(env_name: str, prompt: str, *, secret: bool = False) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    return prompt_value(prompt, secret=secret)


def normalize_prefix(prefix: str) -> str:
    cleaned = prefix.strip().strip("/")
    if not cleaned:
        raise SystemExit("R2 prefix cannot be empty.")
    return cleaned


def list_datasets(viewer_root: Path) -> list[Path]:
    if not viewer_root.exists():
        raise SystemExit(f"Viewer root does not exist: {viewer_root}")
    datasets = [path for path in sorted(viewer_root.iterdir()) if path.is_dir()]
    if not datasets:
        raise SystemExit(f"No dataset directories found under: {viewer_root}")
    return datasets


def select_datasets(viewer_root: Path, requested: list[str] | None) -> list[Path]:
    available = {path.name: path for path in list_datasets(viewer_root)}
    if not requested:
        return list(available.values())

    selected: list[Path] = []
    missing: list[str] = []
    for name in requested:
        dataset = available.get(name)
        if dataset is None:
            missing.append(name)
            continue
        selected.append(dataset)

    if missing:
        names = ", ".join(sorted(available))
        raise SystemExit(f"Unknown dataset(s): {', '.join(missing)}. Available datasets: {names}")
    return selected


def is_publishable_file(path: Path) -> bool:
    return path.suffix.lower() in PUBLISH_EXTENSIONS


def get_cache_control_for_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return NO_CACHE_CONTROL

    if path.name in {"tiles_index.json", "overview_footprints.json"}:
        return NO_CACHE_CONTROL

    if suffix in {".json", ".geojson"}:
        return JSON_CACHE_CONTROL

    return DEFAULT_CACHE_CONTROL


def collect_publish_files(
    html_path: Path,
    viewer_root: Path,
    datasets: list[Path],
    include_root_index: bool,
) -> list[PublishFile]:
    if not html_path.exists():
        raise SystemExit(f"Viewer HTML not found: {html_path}")

    publish_files = [
        PublishFile(
            source=html_path,
            relative_path=Path(html_path.name),
            cache_control=get_cache_control_for_file(html_path),
        )
    ]

    root_index = REPO_ROOT / "tiles_index.json"
    if include_root_index and root_index.exists():
        publish_files.append(
            PublishFile(
                source=root_index,
                relative_path=Path(root_index.name),
                cache_control=get_cache_control_for_file(root_index),
            )
        )

    for dataset in datasets:
        for source in sorted(dataset.rglob("*")):
            if not source.is_file() or not is_publishable_file(source):
                continue
            publish_files.append(
                PublishFile(
                    source=source,
                    relative_path=get_publish_relative_path(source, viewer_root),
                    cache_control=get_cache_control_for_file(source),
                )
            )

    deduped: dict[Path, PublishFile] = {}
    for entry in publish_files:
        deduped[entry.relative_path] = entry
    return list(deduped.values())


def infer_tile_id(tile_dir: Path) -> str | None:
    match = TILE_SUFFIX_PATTERN.search(tile_dir.name)
    if not match:
        return None
    return match.group("tile_id")


def get_publish_relative_path(source: Path, viewer_root: Path) -> Path:
    try:
        return source.relative_to(REPO_ROOT)
    except ValueError:
        try:
            source_relative = source.relative_to(viewer_root)
        except ValueError:
            return Path(source.name)
        return DEFAULT_VIEWER_ROOT_RELATIVE / source_relative


def dataset_has_overview_bundle_assets(dataset: Path) -> bool:
    overview_patterns = (
        "*overview*.tif",
        "*overview*.tiff",
        "*overview*.png",
        "*overview*.jpg",
        "*overview*.jpeg",
        "*overview*.webp",
        "*overview*.json",
    )

    for pattern in overview_patterns:
        if next(dataset.rglob(pattern), None) is not None:
            return True
    return False


def load_dataset_tile_inventory(dataset: Path) -> set[str] | None:
    tiles_index_path = dataset / "tiles_index.json"
    if not tiles_index_path.exists():
        return None

    try:
        payload = json.loads(tiles_index_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    source_tile_dir_value = payload.get("_meta", {}).get("source_tile_dir")
    if not source_tile_dir_value:
        return None

    source_tile_dir = Path(source_tile_dir_value)
    if not source_tile_dir.exists():
        return None

    tile_ids: set[str] = set()
    for path in source_tile_dir.rglob("*"):
        if not path.is_file():
            continue
        match = SOURCE_TILE_PATTERN.search(path.name)
        if match:
            tile_ids.add(match.group("tile_id"))

    return tile_ids or None


def validate_dataset_assets(datasets: list[Path]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    for dataset in datasets:
        expected_tile_ids = load_dataset_tile_inventory(dataset)
        child_dirs = [path for path in sorted(dataset.iterdir()) if path.is_dir()]
        tile_dirs = [path for path in child_dirs if infer_tile_id(path) is not None]
        other_dirs = [path.name for path in child_dirs if infer_tile_id(path) is None]
        has_overview_bundle = dataset_has_overview_bundle_assets(dataset)

        if expected_tile_ids is not None:
            stale_tile_dirs = [path for path in tile_dirs if infer_tile_id(path) not in expected_tile_ids]
            tile_dirs = [path for path in tile_dirs if infer_tile_id(path) in expected_tile_ids]
            if stale_tile_dirs:
                preview = ", ".join(path.name for path in stale_tile_dirs[:3])
                suffix = "..." if len(stale_tile_dirs) > 3 else ""
                warnings.append(
                    f"{dataset.name}: ignored stale tile directories not present in source tile inventory: {preview}{suffix}"
                )

            published_tile_ids = {infer_tile_id(path) for path in tile_dirs if infer_tile_id(path) is not None}
            missing_tile_ids = sorted(expected_tile_ids - published_tile_ids, key=lambda value: int(value))
            for tile_id in missing_tile_ids:
                warnings.append(f"{dataset.name}: missing published tile output for source tile {tile_id}")

        if not tile_dirs:
            if has_overview_bundle:
                continue
            warnings.append(f"{dataset.name}: no tile directories found")
            continue

        if other_dirs and not has_overview_bundle:
            preview = ", ".join(other_dirs[:3])
            suffix = "..." if len(other_dirs) > 3 else ""
            warnings.append(f"{dataset.name}: ignored non-tile directories during validation: {preview}{suffix}")

        for tile_dir in tile_dirs:
            tile_id = infer_tile_id(tile_dir)
            if tile_id is None:
                continue

            has_predictions = any(
                (tile_dir / name).exists()
                for name in (f"predictions_tile_{tile_id}.geojson", "predictions.geojson")
            )
            has_predictor = (tile_dir / f"predictor_tile_{tile_id}_epsg4326_cog.tif").exists()
            has_thumbnail = any(
                (tile_dir / name).exists()
                for name in (f"thumbnail_531_tile_{tile_id}.png", "thumbnail_531.png")
            )
            has_stac = (tile_dir / "stac" / f"item_tile{tile_id}.json").exists()

            if not has_predictions:
                errors.append(f"{dataset.name}/{tile_dir.name}: missing prediction GeoJSON")
            if not has_predictor:
                warnings.append(f"{dataset.name}/{tile_dir.name}: missing predictor COG")
            if not has_thumbnail:
                warnings.append(f"{dataset.name}/{tile_dir.name}: missing thumbnail PNG")
            if not has_stac:
                warnings.append(f"{dataset.name}/{tile_dir.name}: missing STAC item")

    return warnings, errors


def load_boto3_modules():
    try:
        boto3 = importlib.import_module("boto3")
        boto3_exceptions = importlib.import_module("boto3.exceptions")
        config_module = importlib.import_module("botocore.config")
        exceptions_module = importlib.import_module("botocore.exceptions")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "boto3 is required for R2 uploads. Install it with: python -m pip install boto3"
        ) from exc

    return (
        boto3,
        config_module.Config,
        exceptions_module.BotoCoreError,
        exceptions_module.ClientError,
        boto3_exceptions.S3UploadFailedError,
    )


def get_shared_credentials_file() -> Path:
    env_path = os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".aws" / "credentials"


def shared_credentials_profile_exists(profile_name: str | None) -> bool:
    credentials_file = get_shared_credentials_file()
    if not credentials_file.exists():
        return False

    parser = configparser.RawConfigParser()
    parser.read(credentials_file)
    target_profile = (profile_name or os.environ.get("AWS_PROFILE") or "default").strip() or "default"
    return parser.has_section(target_profile)


def build_r2_client(
    account_id: str,
    access_key: str | None = None,
    secret_key: str | None = None,
    profile_name: str | None = None,
):
    boto3, Config, _, _, _ = load_boto3_modules()
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    session_kwargs: dict[str, str] = {}
    if profile_name:
        session_kwargs["profile_name"] = profile_name

    session = boto3.session.Session(**session_kwargs)
    client_kwargs = {
        "endpoint_url": endpoint_url,
        "region_name": "auto",
        "config": Config(signature_version="s3v4"),
    }
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key

    return session.client("s3", **client_kwargs)


def resolve_r2_connection_settings(
    bucket: str,
    profile_name: str | None,
) -> tuple[str, str, str | None, str | None, str | None]:
    resolved_bucket = bucket.strip() if bucket else ""
    if not resolved_bucket:
        resolved_bucket = prompt_value("R2 bucket name: ")

    account_id = resolve_runtime_value("R2_ACCOUNT_ID", "Cloudflare account ID: ")
    normalized_profile = (profile_name or "").strip() or None

    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip() or None
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip() or None

    if normalized_profile and not (access_key and secret_key):
        return resolved_bucket, account_id, None, None, normalized_profile

    if not normalized_profile and not (access_key and secret_key) and shared_credentials_profile_exists(None):
        resolved_profile = (os.environ.get("AWS_PROFILE") or "default").strip() or "default"
        return resolved_bucket, account_id, None, None, resolved_profile

    access_key = resolve_runtime_value(
        "R2_ACCESS_KEY_ID",
        "Cloudflare R2 access key ID (hidden): ",
        secret=True,
    )
    secret_key = resolve_runtime_value(
        "R2_SECRET_ACCESS_KEY",
        "Cloudflare R2 secret access key (hidden): ",
        secret=True,
    )
    return resolved_bucket, account_id, access_key, secret_key, normalized_profile


def resolve_bucket_name(bucket: str) -> str:
    resolved_bucket = bucket.strip() if bucket else ""
    if resolved_bucket:
        return resolved_bucket
    return prompt_value("R2 bucket name: ")


def content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if path.suffix.lower() == ".geojson":
        return "application/geo+json"
    return "application/octet-stream"


def format_upload_failure(
    exc: Exception,
    *,
    bucket: str,
    prefix: str,
    account_id: str,
    profile_name: str | None,
) -> str:
    message = str(exc)
    active_profile = profile_name or os.environ.get("AWS_PROFILE") or "default or environment"

    if "SignatureDoesNotMatch" in message:
        return "\n".join(
            [
                "R2 upload failed with SignatureDoesNotMatch.",
                f"Bucket: {bucket}",
                f"Prefix: {prefix}",
                f"Account ID: {account_id}",
                f"Credentials source: {active_profile}",
                "The access key ID and secret access key being used do not match what Cloudflare R2 expects for this account.",
                "Update ~/.aws/credentials or the selected profile with the correct R2 key pair, or clear any stale AWS_* credential environment variables.",
                "If you are using a saved profile for R2, prefer a dedicated profile such as [r2] and run with --profile r2 or export AWS_PROFILE=r2.",
            ]
        )

    if "AccessDenied" not in message:
        return f"R2 upload failed: {exc}"

    return "\n".join(
        [
            "R2 upload failed with AccessDenied.",
            f"Bucket: {bucket}",
            f"Prefix: {prefix}",
            f"Account ID: {account_id}",
            f"Credentials source: {active_profile}",
            "Check that the R2 API token belongs to this Cloudflare account and has Object Write access to this bucket.",
            "If the bucket name is correct, verify the token was created for the same account and bucket you are uploading to.",
            "If you do not want the extra top-level path segment, use --prefix fvc_composition-viewer.",
        ]
    )


def upload_files(client, bucket: str, prefix: str, files: list[PublishFile], dry_run: bool) -> None:
    for entry in files:
        key = f"{prefix}/{entry.relative_path.as_posix()}"
        print(f"upload {entry.source} -> s3://{bucket}/{key}")
        if dry_run:
            continue

        extra_args = {
            "ContentType": content_type_for(entry.source),
            "CacheControl": entry.cache_control,
        }
        client.upload_file(str(entry.source), bucket, key, ExtraArgs=extra_args)


def main() -> int:
    args = parse_args()

    prefix = normalize_prefix(args.prefix)
    viewer_root = Path(args.viewer_root).resolve()
    html_path = Path(args.html).resolve()
    datasets = select_datasets(viewer_root, args.datasets)
    warnings, errors = validate_dataset_assets(datasets)

    if warnings:
        print("Validation warnings:", file=sys.stderr)
        for message in warnings:
            print(f"  - {message}", file=sys.stderr)

    if errors:
        print("Validation errors:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 2

    if warnings and args.strict:
        print("Strict mode enabled and validation warnings were found.", file=sys.stderr)
        return 2

    files = collect_publish_files(
        html_path=html_path,
        viewer_root=viewer_root,
        datasets=datasets,
        include_root_index=args.include_root_index,
    )
    print(f"Prepared {len(files)} files for upload from {len(datasets)} dataset(s).")

    if args.dry_run:
        dry_run_bucket = (args.bucket or "").strip() or "<bucket>"
        upload_files(None, dry_run_bucket, prefix, files, dry_run=True)
        return 0

    bucket = resolve_bucket_name(args.bucket or "")
    bucket, account_id, access_key, secret_key, profile_name = resolve_r2_connection_settings(
        bucket,
        args.profile,
    )
    _, _, BotoCoreError, ClientError, S3UploadFailedError = load_boto3_modules()

    try:
        client = build_r2_client(account_id, access_key, secret_key, profile_name)
        upload_files(client, bucket, prefix, files, dry_run=False)
    except (BotoCoreError, ClientError, S3UploadFailedError) as exc:
        print(
            format_upload_failure(
                exc,
                bucket=bucket,
                prefix=prefix,
                account_id=account_id,
                profile_name=profile_name,
            ),
            file=sys.stderr,
        )
        return 1

    print(f"Upload complete: s3://{bucket}/{prefix}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())