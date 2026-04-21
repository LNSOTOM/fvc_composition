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

## 4b) Full orthophoto streaming inference

For the full orthophoto, use the streaming mask writer so the prediction runs window-by-window and preserves source nodata as output nodata `255`.

```bash
cd /home/laura/Documents/code/fvc_composition && \
/home/laura/miniconda3/bin/conda run -p /home/laura/.local/share/mamba/envs/fvc_composition \
python -m phase_3_models.unet_site_models.inference_FVCmapping_streaming \
  --variant medium \
  --model-path phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth \
  --input-raster /media/laura/8402326D023263F8/calperumResearch/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD001_dual_ortho_01_bilinear.tif \
  --output-mask phase_3_models/unet_site_models/outputs_full_ortho/SASMDD0001_20220519_medium_epoch55_mask.tif \
  --output-mask-cog phase_3_models/unet_site_models/outputs_full_ortho/SASMDD0001_20220519_medium_epoch55_mask_cog.tif \
  --in-channels 5 \
  --input-bands 2,4,6,8,10 \
  --window-size 256 \
  --overwrite
```

Convert the resulting mask GeoTIFF to GeoJSON polygons with:

```bash
cd /home/laura/Documents/code/fvc_composition && \
/home/laura/miniconda3/bin/conda run -p /home/laura/.local/share/mamba/envs/fvc_composition \
python -m phase_3_models.unet_site_models.mask_geotiff_to_geojson \
  --input-mask phase_3_models/unet_site_models/outputs_full_ortho/SASMDD0001_20220519_medium_epoch55_mask.tif \
  --output-geojson phase_3_models/unet_site_models/outputs_full_ortho/SASMDD0001_20220519_medium_epoch55_mask.geojson \
  --output-crs EPSG:4326 \
  --variant medium \
  --valid-classes 0,1,2,3 \
  --mask-nodata 255

```

If you need a copy in the mask's original CRS for comparison, rerun with:

```bash
python -m phase_3_models.unet_site_models.mask_geotiff_to_geojson \
  --input-mask phase_3_models/unet_site_models/outputs_full_ortho/SASMDD0001_20220519_medium_epoch55_mask.tif \
  --output-geojson phase_3_models/unet_site_models/outputs_full_ortho/SASMDD0001_20220519_medium_epoch55_mask_epsg7854.geojson \
  --output-crs EPSG:7854 \
  --variant medium \
  --valid-classes 0,1,2,3 \
  --mask-nodata 255
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

## Whole orthomosaic streaming inference

Recommended order for a full 10-band orthomosaic:

1. Create a color composite for visual QA / STAC preview.
2. Optionally create a separate 5-band predictor raster with bands `2,4,6,8,10`.
3. Run streaming inference either on the original 10-band orthomosaic using `--input-bands "2,4,6,8,10"`, or on the 5-band predictor raster.

### 1) Create the composite color first

For the original 10-band MicaSense dual orthomosaic, the direct false-color preview is:

```bash
python phase_1_image_processing/step_5_stackBands_predictors/create_composite_colour_cli.py \
  --input "/media/laura/8402326D023263F8/calperumResearch/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD001_dual_ortho_01_bilinear.tif" \
  --output "phase_1_image_processing/step_5_stackBands_predictors/outputs/site1_false_color_1062_cog.tif" \
  --bands "10,6,2" \
  --pmin 1 \
  --pmax 99 \
  --output-driver COG
```

### 2) Optional: create a 5-band predictor raster

This replaces the hard-coded `5_2_imageProcessing_stackedBandsRaster.py` path and keeps the legacy predictor band order `2,4,6,8,10`.

```bash
python phase_1_image_processing/step_5_stackBands_predictors/create_stacked_bands_cli.py \
  --input "/media/laura/8402326D023263F8/calperumResearch/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD001_dual_ortho_01_bilinear.tif" \
  --output "phase_1_image_processing/step_5_stackBands_predictors/outputs/site1_predictor_246810.tif" \
  --bands "2,4,6,8,10" \
  --output-driver GTiff
```

### 3) Run streaming inference

If you want to run inference directly on a full orthomosaic instead of first
splitting it into `tiles_3072`, use:

- `phase_3_models/unet_site_models/inference_FVCmapping_streaming.py`

This script reads one `--window-size` chunk at a time and writes the predicted
class mask directly to disk, so it works on large rasters without holding the
entire orthomosaic in memory.

Example using the site 1 multispectral orthomosaic and the medium checkpoint:

```bash
python phase_3_models/unet_site_models/inference_FVCmapping_streaming.py \
  --variant medium \
  --model-path "phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth" \
  --input-raster "/media/laura/8402326D023263F8/calperumResearch/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD001_dual_ortho_01_bilinear.tif" \
  --output-mask "phase_3_models/unet_site_models/outputs_full_ortho/site1_medium_mask.tif" \
  --output-mask-cog "phase_3_models/unet_site_models/outputs_full_ortho/site1_medium_mask_cog.tif" \
  --window-size 256 \
  --input-bands "2,4,6,8,10" \
  --in-channels 5 \
  --valid-classes "0,1,2,3"
```

Example using the separate 5-band predictor raster instead of the original 10-band orthomosaic:

```bash
python phase_3_models/unet_site_models/inference_FVCmapping_streaming.py \
  --variant medium \
  --model-path "phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth" \
  --input-raster "phase_1_image_processing/step_5_stackBands_predictors/outputs/site1_predictor_246810.tif" \
  --output-mask "phase_3_models/unet_site_models/outputs_full_ortho/site1_medium_mask.tif" \
  --output-mask-cog "phase_3_models/unet_site_models/outputs_full_ortho/site1_medium_mask_cog.tif" \
  --window-size 256 \
  --in-channels 5 \
  --valid-classes "0,1,2,3"
```

Notes:

- The input raster must already be in the predictor band layout expected by the checkpoint.
- For original 10-band MicaSense dual orthomosaics, use `--input-bands "2,4,6,8,10"` to match the legacy predictor selection, or first create a separate 5-band predictor raster.
- `--output-mask-cog` is optional; use it when you want a web/STAC-friendly COG copy.

## Parameterized color composite generation

To build a georeferenced 3-band color composite from a multispectral raster or a
directory of rasters, use:

- `phase_1_image_processing/step_5_stackBands_predictors/create_composite_colour_cli.py`

This script estimates stretch percentiles from a downsampled preview, then
streams the full raster window-by-window. It preserves georeferencing and can
write a Cloud-Optimized GeoTIFF suitable for use as a STAC asset.

Example false-color composite from the original 10-band orthomosaic using bands
`10,6,2`:

```bash
python phase_1_image_processing/step_5_stackBands_predictors/create_composite_colour_cli.py \
  --input "/media/laura/8402326D023263F8/calperumResearch/SASMDD0001/20220519/micasense_dual/level_1/20220519_SASMDD001_dual_ortho_01_bilinear.tif" \
  --output "phase_1_image_processing/step_5_stackBands_predictors/outputs/site1_false_color_cog.tif" \
  --bands "10,6,2" \
  --pmin 1 \
  --pmax 99 \
  --output-driver COG
```

Example batch mode over a directory of predictor tiles:

```bash
python phase_1_image_processing/step_5_stackBands_predictors/create_composite_colour_cli.py \
  --input "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/multispec/res_01/stacked" \
  --output "phase_1_image_processing/step_5_stackBands_predictors/outputs/site1_tiles_false_color" \
  --bands "5,3,1" \
  --output-driver COG
```

Notes:

- Use `10,6,2` for the legacy false-color composite from the original 10-band MicaSense product.
- Use `5,3,1` when the input is already the 5-band stacked predictor raster.
- Directory input writes one output file per input GeoTIFF with the `composite_percentile_` prefix by default.

- `--stage-mode symlink` avoids duplicating large `.tif` inputs, but some drives/mounts don’t allow symlinks (you’ll see “Operation not permitted”). If so, use `--stage-mode copy` (or just re-run: the batch script will fall back to copying automatically).
- Use `--overwrite` if you want to regenerate outputs for tiles that already exist.
- Use `--python` if the batch script is launched in a different environment than the one you want for inference.
