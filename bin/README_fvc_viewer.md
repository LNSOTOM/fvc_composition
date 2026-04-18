# FVC Viewer Cloudflare R2 publish

This README keeps the Cloudflare R2 upload workflow for the FVC viewer next to `upload_viewer_to_r2.py`.

If the viewer data is too large for a Heroku slug, publish the viewer bundle to Cloudflare R2 and point the site route at the public R2 prefix.

Install the extra dependency if your environment predates this change:

```bash
python -m pip install "boto3>=1.35,<2"
```

Set non-sensitive values directly if you want, but do not paste secrets into `export` commands:

Run the following commands from the `fvc_composition` repo root, not from `sotomayorstudio`:

```bash
cd /home/laura/Documents/code/fvc_composition

export R2_ACCOUNT_ID="<cloudflare-account-id>"
export R2_BUCKET="<r2-bucket-name>"  # bucket name only, not https://...r2.cloudflarestorage.com
```

If `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` are not already present in the environment, the uploader now prompts for them interactively and hides both values while you type.

If you do not want to type the key every time, save it once in an AWS-style credentials profile and reuse it:

```bash
mkdir -p ~/.aws
chmod 700 ~/.aws
nano ~/.aws/credentials
```

Add this content and replace the placeholders:

```ini
[r2]
aws_access_key_id = <r2-access-key-id>
aws_secret_access_key = <r2-secret-access-key>
```

If you prefer, you can use the default profile instead:

```ini
[default]
aws_access_key_id = <r2-access-key-id>
aws_secret_access_key = <r2-secret-access-key>
```

Then protect the file and run the uploader with that profile:

```bash
chmod 600 ~/.aws/credentials

cd /home/laura/Documents/code/fvc_composition
export R2_ACCOUNT_ID="<cloudflare-account-id>"
export R2_BUCKET="<r2-bucket-name>"  # bucket name only, not https://...r2.cloudflarestorage.com

python bin/upload_viewer_to_r2.py \
  --profile r2 \
  --dataset medium_multispec5b
```

Exact command used in this workspace environment:

```bash
export AWS_PROFILE=r2 && export R2_ACCOUNT_ID=6e92132427037064d5167bcac10a15b5 && export R2_BUCKET=terrascientia && /home/laura/miniconda3/bin/conda run -p /home/laura/.local/share/mamba/envs/fvc_composition --no-capture-output python /home/laura/.vscode-server/extensions/ms-python.python-2026.5.2026032701-linux-x64/python_files/get_output_via_markers.py bin/upload_viewer_to_r2.py --profile r2 --bucket terrascientia --prefix fvc_composition-viewer --dataset medium_multispec5b
```

If you want that profile used automatically in future shells, set:

```bash
export AWS_PROFILE=r2
```

If your credentials are stored under the default profile in `~/.aws/credentials`, you do not need `--profile` or `AWS_PROFILE`; the uploader will use that saved profile automatically and will stop prompting for the access key and secret.

If you want to keep using environment variables in the current shell, use hidden prompts instead of pasting the secrets:

```bash
cd /home/laura/Documents/code/fvc_composition

export R2_ACCOUNT_ID="<cloudflare-account-id>"
export R2_BUCKET="<r2-bucket-name>"  # bucket name only, not https://...r2.cloudflarestorage.com

read -rsp "Cloudflare R2 access key ID: " R2_ACCESS_KEY_ID; echo
export R2_ACCESS_KEY_ID

read -rsp "Cloudflare R2 secret access key: " R2_SECRET_ACCESS_KEY; echo
export R2_SECRET_ACCESS_KEY

python bin/upload_viewer_to_r2.py --dataset medium_multispec5b

unset R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
```

The default object prefix is:

```text
terrascientia/fvc_composition-viewer/
```

If your bucket itself is named `terrascientia`, the resulting object path will be:

```text
s3://terrascientia/terrascientia/fvc_composition-viewer/...
```

That is valid, but if you want the objects directly under `s3://terrascientia/fvc_composition-viewer/...`, override the prefix explicitly:

```bash
cd /home/laura/Documents/code/fvc_composition

python bin/upload_viewer_to_r2.py \
  --bucket terrascientia \
  --prefix fvc_composition-viewer \
  --dataset medium_multispec5b
```

Dry-run the publish first so you can see exactly which files will be uploaded:

```bash
cd /home/laura/Documents/code/fvc_composition

python bin/upload_viewer_to_r2.py \
  --dataset medium_multispec5b \
  --dry-run
```

Upload the selected dataset:

```bash
cd /home/laura/Documents/code/fvc_composition

python bin/generate_tiles_index.py medium_multispec5b

python bin/upload_viewer_to_r2.py \
  --dataset medium_multispec5b
```

If a dataset is prediction-only and does not yet have COG, thumbnail, or STAC files, generate its dataset-local `tiles_index.json` first so the viewer can still populate the tile selector from the available GeoJSON folders.

If you want the real map image and thumbnail instead of prediction polygons only, the next step is to run the generation workflow from the original predictor rasters so the missing COG, PNG thumbnail, and STAC files are created before upload. See `INFERENCE_WORKFLOW.md` for the source-raster workflow.

For the most secure interactive flow, skip the credential exports entirely and let the script prompt you:

```bash
cd /home/laura/Documents/code/fvc_composition

python bin/upload_viewer_to_r2.py \
  --bucket <r2-bucket-name> \
  --dataset medium_multispec5b
```

Upload multiple datasets if needed:

```bash
cd /home/laura/Documents/code/fvc_composition

python bin/upload_viewer_to_r2.py \
  --dataset medium_multispec5b \
  --dataset medium \
  --dataset low
```

Notes:

- The uploader publishes only web-facing files: `cnn_mappingAI_viewer.html`, `.json`, `.geojson`, `.tif`, `.tiff`, and image assets. It skips shapefile sidecars because the viewer does not load them.
- The script validates a sample of each dataset before upload and warns when expected viewer assets such as COGs, STAC items, or thumbnails are missing.
- Use `--strict` if you want the upload to fail on those warnings.

After the R2 bucket is public, your public viewer prefix will look like:

```text
https://<public-r2-host>/terrascientia/fvc_composition-viewer/
```

The portfolio site route in `sotomayorstudio` can proxy that prefix directly. Set this config var on the `lauransotomayor` app:

```bash
FVC_COMPOSITION_VIEWER_UPSTREAM_URL=https://<public-r2-host>/terrascientia/fvc_composition-viewer
```

Then `/fvc_composition-viewer/` on the main site will serve the R2-hosted viewer through the existing Flask proxy.