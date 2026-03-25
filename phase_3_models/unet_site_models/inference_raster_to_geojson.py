"""Run tiled U-Net inference and export polygons to GeoJSON.

This script is inference-only and intentionally does NOT import `config_param.py`.
`config_param.py` contains training-side effects (e.g. directory creation / mask scans),
which can make inference brittle.

Use `--variant` presets to match class label conventions:
- low:    BE=0, NPV=1, PV=2
- low_sparse: BE=0, NPV=1, PV=2 (sparse sampling)
- medium: BE=0, NPV=1, PV=2, SI=3
- medium_sparse: BE=0, NPV=1, PV=2, SI=3 (medium class scheme on sparse sampling)
- dense:  BE=0, NPV=1, PV=2, SI=3, WI=4
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes, sieve
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm

from phase_3_models.unet_site_models.model.unet_model_architecture_inference import UNetModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VARIANT_PRESETS: dict[str, dict] = {
    "low": {
        "class_labels": {"BE": 0, "NPV": 1, "PV": 2},
    },
    "low_sparse": {
        # Sparse sampling, but using the low 3-class scheme.
        "class_labels": {"BE": 0, "NPV": 1, "PV": 2},
    },
    "medium": {
        "class_labels": {"BE": 0, "NPV": 1, "PV": 2, "SI": 3},
    },
    "medium_sparse": {
        # Low sampling density, but using the 4-class scheme compatible with the medium checkpoint.
        "class_labels": {"BE": 0, "NPV": 1, "PV": 2, "SI": 3},
    },
    "dense": {
        "class_labels": {"BE": 0, "NPV": 1, "PV": 2, "SI": 3, "WI": 4},
    },
}


DEFAULT_TILE_SIZE = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_POSITIVE_CLASS_ID = 1
DEFAULT_INCLUDE_CLASS_ZERO = True
DEFAULT_MIN_POLYGON_AREA = 0.0
DEFAULT_SIEVE_SIZE = 0
DEFAULT_SI_MIN_POLYGON_AREA = 0.0
DEFAULT_SI_MAX_POLYGON_AREA = 0.0

DEFAULT_IN_CHANNELS = 5


def parse_class_id_map(raw: Optional[str]) -> dict[int, int]:
    raw = (raw or "").strip()
    if not raw:
        return {}

    mapping: dict[int, int] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(
                "--class-id-map must be comma-separated 'src:dst' pairs, e.g. '1:2,2:1'"
            )
        src_s, dst_s = [x.strip() for x in part.split(":", 1)]
        src = int(src_s)
        dst = int(dst_s)
        if src in mapping:
            raise ValueError(f"Duplicate mapping for class id {src} in --class-id-map")
        mapping[src] = dst
    return mapping


def apply_class_id_map(mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    if not mapping:
        return mask

    if mask.dtype.kind not in {"u", "i"}:
        mask = mask.astype(np.int32, copy=False)

    max_seen = int(np.max(mask)) if mask.size else 0
    max_key = max(mapping.keys(), default=0)
    max_val = max(mapping.values(), default=0)
    size = max(max_seen, max_key, max_val) + 1

    lut = np.arange(size, dtype=mask.dtype)
    for src, dst in mapping.items():
        if src < 0 or dst < 0:
            raise ValueError("--class-id-map does not support negative class ids")
        if src >= size:
            # Expand LUT if needed
            new_size = src + 1
            new_lut = np.arange(new_size, dtype=lut.dtype)
            new_lut[: lut.size] = lut
            lut = new_lut
        lut[src] = dst

    # If mask contains values beyond lut, expand once more.
    if int(np.max(mask)) >= lut.size:
        new_size = int(np.max(mask)) + 1
        new_lut = np.arange(new_size, dtype=lut.dtype)
        new_lut[: lut.size] = lut
        lut = new_lut

    return lut[mask]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled U-Net inference and export polygons to GeoJSON")

    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_PRESETS.keys()),
        default="medium",
        help=(
            "Inference preset controlling default classes/out-channels "
            "(default: medium)"
        ),
    )

    parser.add_argument("--model-path", default="", help="Path to a .pth checkpoint")
    parser.add_argument(
        "--input-raster",
        default="",
        help="Path to input .tif OR a directory containing .tif tiles",
    )
    parser.add_argument(
        "--output-geojson",
        default="",
        help="Output .geojson file (single input) OR output directory (directory input)",
    )

    parser.add_argument(
        "--output-mask",
        default="",
        help=(
            "Optional output class-mask GeoTIFF path (single input) OR output directory (directory input). "
            "If provided, writes a single-band uint8 raster with predicted class ids."
        ),
    )

    parser.add_argument(
        "--output-shp",
        default="",
        help=(
            "Optional output ESRI Shapefile .shp path (single input) OR output directory (directory input). "
            "Writes polygons equivalent to the GeoJSON output."
        ),
    )

    parser.add_argument(
        "--tile-id",
        default="",
        help=(
            "Optional tile id to attach to each GeoJSON feature as properties.tile_id and properties.tile_number. "
            "Useful when running per-tile outputs via batch scripts."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--positive-class-id", type=int, default=DEFAULT_POSITIVE_CLASS_ID)
    parser.add_argument("--include-class-zero", action="store_true", default=DEFAULT_INCLUDE_CLASS_ZERO)
    parser.add_argument("--exclude-class-zero", action="store_false", dest="include_class_zero")
    parser.add_argument("--min-polygon-area", type=float, default=DEFAULT_MIN_POLYGON_AREA)
    parser.add_argument(
        "--valid-classes",
        type=str,
        default=None,
        help=(
            "Comma-separated class ids. If omitted, defaults based on --variant "
            "(low: 0,1,2; medium: 0,1,2,3; dense: 0,1,2,3,4)."
        ),
    )
    parser.add_argument("--sieve-size", type=int, default=DEFAULT_SIEVE_SIZE)
    parser.add_argument("--si-min-polygon-area", type=float, default=DEFAULT_SI_MIN_POLYGON_AREA)
    parser.add_argument("--si-max-polygon-area", type=float, default=DEFAULT_SI_MAX_POLYGON_AREA)
    parser.add_argument(
        "--in-channels",
        type=int,
        default=DEFAULT_IN_CHANNELS,
        help="Model input channels; if omitted, inferred from checkpoint when possible",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=None,
        help=(
            "Model output channels; if omitted, inferred from checkpoint when possible, "
            "otherwise defaults to number of classes in --variant"
        ),
    )

    parser.add_argument(
        "--class-id-map",
        type=str,
        default="",
        help=(
            "Optional class id remapping applied after prediction, before polygonization. "
            "Format: 'src:dst,src2:dst2'. Example to swap NPV/PV: '1:2,2:1'."
        ),
    )
    return parser.parse_args()


def parse_valid_classes(raw_value):
    classes = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        classes.append(int(part))

    if not classes:
        raise ValueError("--valid-classes must contain at least one class id, e.g. '0,1,2,3'.")

    return sorted(set(classes))


def _normalize_state_dict(state_dict: dict) -> dict:
    # Common checkpoint formats:
    # - torch.save(UNetModule().state_dict())  -> keys start with "model."
    # - Lightning checkpoints                  -> may have top-level "state_dict"
    # We want keys that match UNetModel ("downs...", "final_conv...")
    if any(k.startswith("model.") for k in state_dict.keys()):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _infer_channels_from_state_dict(state_dict: dict) -> tuple[Optional[int], Optional[int]]:
    in_ch: Optional[int] = None
    out_ch: Optional[int] = None

    w0 = state_dict.get("downs.0.conv.0.weight")
    if isinstance(w0, torch.Tensor) and w0.ndim == 4:
        in_ch = int(w0.shape[1])

    w_last = state_dict.get("final_conv.weight")
    if isinstance(w_last, torch.Tensor) and w_last.ndim == 4:
        out_ch = int(w_last.shape[0])

    return in_ch, out_ch


def load_model(model_path: str, in_channels: int, out_channels: int) -> tuple[torch.nn.Module, int, int]:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    raw_state = checkpoint.get("state_dict", checkpoint)
    state_dict = _normalize_state_dict(raw_state)

    inferred_in, inferred_out = _infer_channels_from_state_dict(state_dict)
    resolved_in = inferred_in if inferred_in is not None else int(in_channels)
    resolved_out = inferred_out if inferred_out is not None else int(out_channels)

    model = UNetModel(in_channels=resolved_in, out_channels=resolved_out)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, resolved_in, resolved_out


def preprocess(image: np.ndarray, in_channels: int) -> torch.Tensor:
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if image.ndim != 3:
        raise ValueError(f"Expected tile with 3 dimensions [C,H,W] or [H,W,C], got shape {image.shape}")

    if image.shape[0] != in_channels and image.shape[-1] == in_channels:
        image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image).unsqueeze(0)


@torch.no_grad()
def predict_tile(model: torch.nn.Module, tile: np.ndarray, in_channels: int) -> np.ndarray:
    tensor = preprocess(tile, in_channels=in_channels).to(DEVICE)
    output = model(tensor)

    if output.shape[1] > 1:
        class_map = torch.argmax(output, dim=1)
        return class_map.squeeze(0).cpu().numpy().astype(np.uint8)

    if output.min() < 0.0 or output.max() > 1.0:
        prob = torch.sigmoid(output)
    else:
        prob = output

    mask = (prob > predict_tile.threshold).float()
    return mask.squeeze().cpu().numpy().astype(np.uint8)


predict_tile.threshold = DEFAULT_THRESHOLD
predict_tile.positive_class_id = DEFAULT_POSITIVE_CLASS_ID


def run_inference(model: torch.nn.Module, input_raster: str, tile_size: int, in_channels: int):
    with rasterio.open(input_raster) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

        if src.count < in_channels:
            raise ValueError(
                f"Input raster has {src.count} bands but model expects {in_channels} channels."
            )

        band_indices = list(range(1, in_channels + 1))

        full_mask = np.zeros((height, width), dtype=np.uint8)

        for y in tqdm(range(0, height, tile_size)):
            for x in range(0, width, tile_size):
                win_w = min(tile_size, width - x)
                win_h = min(tile_size, height - y)
                window = Window(x, y, win_w, win_h)

                tile = src.read(band_indices, window=window)

                pred = predict_tile(model, tile, in_channels=in_channels)
                h, w = pred.shape
                full_mask[y:y + h, x:x + w] = pred

    return full_mask, transform, crs


def _iter_input_rasters(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_dir():
        return [str(x) for x in sorted(p.glob("*.tif"))]
    return [str(p)]


def _resolve_output_path(output_geojson: str, input_raster: str, is_batch: bool) -> str:
    out = Path(output_geojson)
    if is_batch:
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_raster).stem
        return str(out / f"{stem}_predictions.geojson")
    # single input: output is a file path
    if out.suffix.lower() != ".geojson":
        # allow user to pass a directory even for single input
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_raster).stem
        return str(out / f"{stem}_predictions.geojson")
    parent = out.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def _resolve_output_mask_path(output_mask: str, input_raster: str, is_batch: bool) -> str:
    out = Path(output_mask)
    if is_batch:
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_raster).stem
        return str(out / f"{stem}_mask.tif")

    if not out.suffix:
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_raster).stem
        return str(out / f"{stem}_mask.tif")

    parent = out.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def _resolve_output_shp_path(output_shp: str, input_raster: str, is_batch: bool) -> str:
    out = Path(output_shp)
    if is_batch:
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_raster).stem
        return str(out / f"{stem}_predictions.shp")

    if out.is_dir() or (not out.suffix):
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_raster).stem
        return str(out / f"{stem}_predictions.shp")

    parent = out.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def save_mask_geotiff(mask: np.ndarray, transform, crs, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
        "predictor": 2,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8, copy=False), 1)


def mask_to_polygons(
    mask,
    transform,
    crs,
    include_class_zero=True,
    min_polygon_area=0.0,
    valid_classes=None,
    sieve_size=0,
    si_min_polygon_area=0.0,
    si_max_polygon_area=0.0,
    class_id_to_name: Optional[dict[int, str]] = None,
):
    features = []

    if valid_classes is None:
        valid_classes = sorted(int(v) for v in np.unique(mask))
    else:
        valid_classes = sorted(set(int(v) for v in valid_classes))

    if not include_class_zero:
        valid_classes = [class_id for class_id in valid_classes if class_id != 0]

    if not valid_classes:
        return gpd.GeoDataFrame({"class_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)

    class_feature_counts = {}

    for class_id in valid_classes:
        class_mask = (mask == class_id).astype(np.uint8)

        if sieve_size > 0:
            class_mask = sieve(class_mask, size=sieve_size, connectivity=8).astype(np.uint8)

        if not np.any(class_mask):
            continue

        current_min_area = max(min_polygon_area, si_min_polygon_area if class_id == 3 else 0.0)

        for geom, _ in shapes(class_mask, mask=class_mask.astype(bool), transform=transform):
            geometry = shape(geom)
            area = geometry.area

            if current_min_area > 0 and area <= current_min_area:
                continue

            if class_id == 3 and si_max_polygon_area > 0 and area >= si_max_polygon_area:
                continue

            features.append({
                "geometry": geometry,
                "class_id": int(class_id),
                **(
                    {"class_name": class_id_to_name.get(int(class_id), f"Unknown-{class_id}")}
                    if isinstance(class_id_to_name, dict)
                    else {}
                ),
            })
            class_feature_counts[class_id] = class_feature_counts.get(class_id, 0) + 1

    if class_feature_counts:
        print("Polygon features by class:", class_feature_counts)

    if not features:
        return gpd.GeoDataFrame({"class_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)

    gdf = gpd.GeoDataFrame(features, crs=crs)

    if min_polygon_area > 0:
        gdf = gdf[gdf.geometry.area > min_polygon_area]

    return gdf


if __name__ == "__main__":
    args = parse_args()

    if not args.model_path:
        raise SystemExit("--model-path is required")
    if not args.input_raster:
        raise SystemExit("--input-raster is required")
    if not args.output_geojson:
        raise SystemExit("--output-geojson is required")

    predict_tile.threshold = args.threshold
    predict_tile.positive_class_id = args.positive_class_id

    preset = VARIANT_PRESETS[args.variant]
    class_id_to_name = {int(v): str(k) for k, v in preset["class_labels"].items()}
    preset_valid = ",".join(str(v) for v in sorted(preset["class_labels"].values()))
    valid_classes = parse_valid_classes(args.valid_classes if args.valid_classes is not None else preset_valid)
    out_channels = int(args.out_channels) if args.out_channels is not None else len(valid_classes)

    class_id_map = parse_class_id_map(args.class_id_map)

    # NOTE: --class-id-map is intended to convert *model output ids* into the canonical ids
    # used by the chosen --variant (and your viewer/QGIS styling). Therefore we remap only
    # the predicted mask values. We intentionally do NOT remap `valid_classes` nor
    # `class_id_to_name`.

    model, resolved_in_channels, resolved_out_channels = load_model(
        args.model_path,
        in_channels=int(args.in_channels),
        out_channels=out_channels,
    )

    input_rasters = _iter_input_rasters(args.input_raster)
    is_batch = Path(args.input_raster).is_dir()

    if is_batch and Path(args.output_geojson).suffix.lower() == ".geojson":
        raise SystemExit(
            "When --input-raster is a directory, --output-geojson must be a directory too."
        )

    for raster_path in input_rasters:
        out_geojson = _resolve_output_path(args.output_geojson, raster_path, is_batch=is_batch)
        out_mask = ""
        if str(args.output_mask).strip():
            out_mask = _resolve_output_mask_path(args.output_mask, raster_path, is_batch=is_batch)
        out_shp = ""
        if str(args.output_shp).strip():
            out_shp = _resolve_output_shp_path(args.output_shp, raster_path, is_batch=is_batch)

        print(f"Running inference on: {raster_path}")
        mask, transform, crs = run_inference(
            model,
            input_raster=raster_path,
            tile_size=args.tile_size,
            in_channels=resolved_in_channels,
        )

        if class_id_map:
            mask = apply_class_id_map(mask, class_id_map)

        if out_mask:
            print(f"Saving mask GeoTIFF -> {out_mask}")
            save_mask_geotiff(mask, transform, crs, out_mask)

        print("Converting to polygons...")
        invalid_present = sorted(int(v) for v in np.unique(mask) if int(v) not in valid_classes)
        if invalid_present:
            print(
                f"Warning: mask contains class values outside --valid-classes {valid_classes}: {invalid_present}"
            )

        gdf = mask_to_polygons(
            mask,
            transform,
            crs,
            include_class_zero=args.include_class_zero,
            min_polygon_area=args.min_polygon_area,
            valid_classes=valid_classes,
            sieve_size=args.sieve_size,
            si_min_polygon_area=args.si_min_polygon_area,
            si_max_polygon_area=args.si_max_polygon_area,
            class_id_to_name=class_id_to_name,
        )

        if not gdf.empty and gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        tile_id_raw = str(args.tile_id).strip()
        if tile_id_raw:
            gdf["tile_id"] = tile_id_raw
            try:
                gdf["tile_number"] = int(tile_id_raw)
            except Exception:
                gdf["tile_number"] = tile_id_raw

        unique_values, counts = np.unique(mask, return_counts=True)
        print("Mask classes (value:count):", dict(zip(unique_values.tolist(), counts.tolist())))
        print(f"Polygon features: {len(gdf)}")

        print(f"Saving GeoJSON -> {out_geojson}")
        gdf.to_file(out_geojson, driver="GeoJSON")

        if out_shp:
            print(f"Saving Shapefile -> {out_shp}")
            gdf.to_file(out_shp, driver="ESRI Shapefile")

    print("Done!")
