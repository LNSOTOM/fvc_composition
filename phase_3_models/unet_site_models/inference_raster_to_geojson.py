import os
import argparse

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes, sieve
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm

from phase_3_models.unet_site_models.model.unet_module import UNetModule
from phase_3_models.unet_site_models import config_param

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = getattr(config_param, "INFERENCE_MODEL_PATH", "models/best_model.ckpt")
INPUT_RASTER = getattr(config_param, "INFERENCE_INPUT_RASTER", "data/orthomosaic.tif")
OUTPUT_GEOJSON = getattr(config_param, "INFERENCE_OUTPUT_GEOJSON", "outputs/predictions.geojson")

TILE_SIZE = getattr(config_param, "INFERENCE_TILE_SIZE", 512)
THRESHOLD = getattr(config_param, "INFERENCE_THRESHOLD", 0.5)
POSITIVE_CLASS_ID = getattr(config_param, "INFERENCE_POSITIVE_CLASS_ID", 1)
INCLUDE_CLASS_ZERO = getattr(config_param, "INFERENCE_INCLUDE_CLASS_ZERO", True)
MIN_POLYGON_AREA = getattr(config_param, "INFERENCE_MIN_POLYGON_AREA", 0.0)
VALID_CLASSES = getattr(config_param, "INFERENCE_VALID_CLASSES", "0,1,2,3")
SIEVE_SIZE = getattr(config_param, "INFERENCE_SIEVE_SIZE", 0)
SI_MIN_POLYGON_AREA = getattr(config_param, "INFERENCE_SI_MIN_POLYGON_AREA", 0.0)
SI_MAX_POLYGON_AREA = getattr(config_param, "INFERENCE_SI_MAX_POLYGON_AREA", 0.0)

MODEL_IN_CHANNELS = getattr(config_param, "IN_CHANNELS", 3)
MODEL_OUT_CHANNELS = getattr(config_param, "OUT_CHANNELS", 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run tiled U-Net inference and export polygons to GeoJSON")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input-raster", default=INPUT_RASTER)
    parser.add_argument("--output-geojson", default=OUTPUT_GEOJSON)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--positive-class-id", type=int, default=POSITIVE_CLASS_ID)
    parser.add_argument("--include-class-zero", action="store_true", default=INCLUDE_CLASS_ZERO)
    parser.add_argument("--exclude-class-zero", action="store_false", dest="include_class_zero")
    parser.add_argument("--min-polygon-area", type=float, default=MIN_POLYGON_AREA)
    parser.add_argument("--valid-classes", type=str, default=VALID_CLASSES)
    parser.add_argument("--sieve-size", type=int, default=SIEVE_SIZE)
    parser.add_argument("--si-min-polygon-area", type=float, default=SI_MIN_POLYGON_AREA)
    parser.add_argument("--si-max-polygon-area", type=float, default=SI_MAX_POLYGON_AREA)
    parser.add_argument("--in-channels", type=int, default=MODEL_IN_CHANNELS)
    parser.add_argument("--out-channels", type=int, default=MODEL_OUT_CHANNELS)
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


def load_model(model_path, in_channels, out_channels):
    model = UNetModule()
    checkpoint = torch.load(model_path, map_location=DEVICE)

    state_dict = checkpoint.get("state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        adapted_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.model.load_state_dict(adapted_state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def preprocess(image):
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if image.ndim != 3:
        raise ValueError(f"Expected tile with 3 dimensions [C,H,W] or [H,W,C], got shape {image.shape}")

    if image.shape[0] != MODEL_IN_CHANNELS and image.shape[-1] == MODEL_IN_CHANNELS:
        image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image).unsqueeze(0)


@torch.no_grad()
def predict_tile(model, tile):
    tensor = preprocess(tile).to(DEVICE)
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


predict_tile.threshold = THRESHOLD
predict_tile.positive_class_id = POSITIVE_CLASS_ID


def run_inference(model, input_raster, tile_size, in_channels):
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

                pred = predict_tile(model, tile)
                h, w = pred.shape
                full_mask[y:y + h, x:x + w] = pred

    return full_mask, transform, crs


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
                "class_id": class_id,
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

    predict_tile.threshold = args.threshold
    predict_tile.positive_class_id = args.positive_class_id

    output_dir = os.path.dirname(args.output_geojson)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = load_model(args.model_path, args.in_channels, args.out_channels)

    print("Running inference...")
    mask, transform, crs = run_inference(
        model,
        input_raster=args.input_raster,
        tile_size=args.tile_size,
        in_channels=args.in_channels,
    )

    print("Converting to polygons...")
    valid_classes = parse_valid_classes(args.valid_classes)

    invalid_present = sorted(int(v) for v in np.unique(mask) if int(v) not in valid_classes)
    if invalid_present:
        print(f"Warning: mask contains class values outside --valid-classes {valid_classes}: {invalid_present}")

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
    )

    if not gdf.empty and gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    unique_values, counts = np.unique(mask, return_counts=True)
    print("Mask classes (value:count):", dict(zip(unique_values.tolist(), counts.tolist())))
    print(f"Polygon features: {len(gdf)}")

    print("Saving GeoJSON...")
    gdf.to_file(args.output_geojson, driver="GeoJSON")

    print("Done!")
