# Tile inference workflow (tiles 22 / 55)

This repo contains a small end-to-end workflow to:

1. Prepare a web-friendly predictor raster (EPSG:4326 Cloud-Optimized GeoTIFF)
2. Run model inference and export prediction polygons to GeoJSON
3. Generate a small PNG thumbnail for the viewer sidebar
4. Generate STAC metadata (`stac/`) so the viewer can resolve assets via STAC
5. View everything in `cnn_mappingAI_viewer.html`

## Prerequisites

- Conda env from `environment.yml` (or an equivalent environment with the repo dependencies)
- GDAL CLI tools available: `gdalwarp`, `gdal_translate`

The viewer requires HTTP `Range` support for COG streaming.

## 1) Prepare the output folders

From the repo root:

```bash
mkdir -p phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22
mkdir -p phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55
```

Copy raw predictors (example input paths):

```bash
cp -f \
  "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.22.tif" \
  "phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22/predictor_tile_22.tif"

cp -f \
  "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.55.tif" \
  "phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55/predictor_tile_55.tif"
```

## 2) Reproject to EPSG:4326

```bash
# Tile 22
out22="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22"
gdalwarp -overwrite -t_srs EPSG:4326 -r bilinear -multi -wo NUM_THREADS=ALL_CPUS \
  "$out22/predictor_tile_22.tif" \
  "$out22/predictor_tile_22_epsg4326.tif"

# Tile 55
out55="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55"
gdalwarp -overwrite -t_srs EPSG:4326 -r bilinear -multi -wo NUM_THREADS=ALL_CPUS \
  "$out55/predictor_tile_55.tif" \
  "$out55/predictor_tile_55_epsg4326.tif"
```

## 3) Convert to a Cloud-Optimized GeoTIFF (COG)

```bash
# Tile 22
out22="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22"
gdal_translate -of COG \
  -co COMPRESS=DEFLATE -co LEVEL=9 \
  -co PREDICTOR=FLOATING_POINT \
  -co OVERVIEW_COMPRESS=DEFLATE -co OVERVIEW_PREDICTOR=FLOATING_POINT \
  -co RESAMPLING=BILINEAR -co OVERVIEWS=AUTO \
  -co NUM_THREADS=ALL_CPUS -co BIGTIFF=IF_SAFER \
  "$out22/predictor_tile_22_epsg4326.tif" \
  "$out22/predictor_tile_22_epsg4326_cog.tif"

# Tile 55
out55="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55"
gdal_translate -of COG \
  -co COMPRESS=DEFLATE -co LEVEL=9 \
  -co PREDICTOR=FLOATING_POINT \
  -co OVERVIEW_COMPRESS=DEFLATE -co OVERVIEW_PREDICTOR=FLOATING_POINT \
  -co RESAMPLING=BILINEAR -co OVERVIEWS=AUTO \
  -co NUM_THREADS=ALL_CPUS -co BIGTIFF=IF_SAFER \
  "$out55/predictor_tile_55_epsg4326.tif" \
  "$out55/predictor_tile_55_epsg4326_cog.tif"
```

## 4) Run inference and export polygons to GeoJSON

This uses `phase_3_models/unet_site_models/inference_raster_to_geojson.py`.

Example using the medium model checkpoint:

```bash
model="phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth"

# Tile 22
out22="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22"
python phase_3_models/unet_site_models/inference_raster_to_geojson.py \
  --model-path "$model" \
  --input-raster "$out22/predictor_tile_22.tif" \
  --output-geojson "$out22/predictions.geojson" \
  --in-channels 5 \
  --valid-classes "0,1,2,3"

# Tile 55
out55="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55"
python phase_3_models/unet_site_models/inference_raster_to_geojson.py \
  --model-path "$model" \
  --input-raster "$out55/predictor_tile_55.tif" \
  --output-geojson "$out55/predictions.geojson" \
  --in-channels 5 \
  --valid-classes "0,1,2,3"
```

## 5) Create sidebar thumbnails

This creates a small PNG preview of the predictor (false-color 5-3-1 by default).

```bash
out22="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22"
python bin/generate_tile_thumbnail.py \
  --input "$out22/predictor_tile_22_epsg4326_cog.tif" \
  --output "$out22/thumbnail_531.png" \
  --size 240

out55="phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55"
python bin/generate_tile_thumbnail.py \
  --input "$out55/predictor_tile_55_epsg4326_cog.tif" \
  --output "$out55/thumbnail_531.png" \
  --size 240
```

## 6) Generate STAC metadata

The viewer can resolve predictor/predictions via STAC (`stac/item_tileXX.json`).

```bash
python bin/generate_tile_stac.py \
  --tile-id 22 \
  --tile-dir phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile22

python bin/generate_tile_stac.py \
  --tile-id 55 \
  --tile-dir phase_3_models/unet_site_models/wombat_predictions_stitch_medium_1024_120ep_raw_bestmodel_55_tile55
```

## 7) View in the browser

Run the Range-capable server (required for COG streaming):

```bash
python3 bin/range_http_server.py 8001
```

Open:

- <http://127.0.0.1:8001/cnn_mappingAI_viewer.html>

Use the **Tile** dropdown in the sidebar to switch between tile 22 and tile 55.

## Batch mode (run all tiles, e.g. ~65)

If you have a folder with many predictor tiles named like `tiles_multispectral.<id>.tif`, you can run the full pipeline for *all* tiles in one command.

This will create one output folder per tile under `phase_3_models/unet_site_models/` (or under `--output-root` if provided) and it will also write `tiles_index.json` at the repo root so the viewer automatically lists all available tiles.

```bash
python bin/batch_inference_tiles.py \
  --input-dir "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked" \
  --model-path "phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth" \
  --in-channels 5 \
  --valid-classes "0,1,2,3" \
  --stage-mode symlink \
  --continue-on-error
```

```bash
python bin/batch_inference_tiles.py \
  --input-dir "//media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/multispec/res_01/stacked" \
  --output-root "phase_3_models/unet_site_models/wombat_mappingAI_viewer/medium_multispec5b" \
  --model-path "phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth" \
  --in-channels 5 \
  --valid-classes "0,1,2,3" \
  --stage-mode none \
  --cleanup-intermediates \
  --write-mask-tif \
  --write-shp \
  --continue-on-error
```

```bash
python bin/batch_inference_tiles.py \
  --input-dir "/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/multispec/res_01/stacked" \
  --output-root "phase_3_models/unet_site_models/wombat_mappingAI_viewer/lowModel_multispec5b" \
  --variant low \
  --model-path "phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_3_epoch_108.pth" \
  --in-channels 5 \
  --valid-classes "0,1,2" \
  --stage-mode none \
  --cleanup-intermediates \
  --write-mask-tif \
  --write-shp \
  --continue-on-error
```

If you see an error like `ModuleNotFoundError: No module named 'torch'`, you’re running with a Python interpreter that doesn’t have the ML deps (often `/usr/bin/python`). Run from the conda env, or pass `--python`:

```bash
# Example: force the interpreter used for inference/thumbnail/STAC steps
python bin/batch_inference_tiles.py \
  --python "$(which python)" \
  --input-dir "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked" \
  --continue-on-error
```

Optional: run only a subset of tiles:

```bash
python bin/batch_inference_tiles.py \
  --input-dir "/path/to/stacked" \
  --tile-ids "22,55" \
  --continue-on-error
```

Notes:

- `--stage-mode symlink` avoids duplicating large `.tif` inputs, but some drives/mounts don’t allow symlinks (you’ll see “Operation not permitted”). If so, use `--stage-mode copy` (or just re-run: the batch script will fall back to copying automatically).
- Use `--overwrite` if you want to regenerate outputs for tiles that already exist.
- Use `--python` if the batch script is launched in a different environment than the one you want for inference.
