# Reproducible runtime for the whole repo (viewer + inference + batch scripts)
# Pattern inspired by research repos like ForestFormer3D: build deps into an image,
# then mount the workspace at runtime so you can iterate without rebuilding.

FROM mambaorg/micromamba:1.5.8

# Keep installs deterministic and layer-cacheable
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Create the conda env defined by environment.yml (name: fvc_composition)
RUN micromamba create -y -f /tmp/environment.yml \
  && micromamba clean --all --yes

WORKDIR /workspace

# Default: run the Range-capable viewer server
EXPOSE 8001
CMD ["micromamba", "run", "-n", "fvc_composition", "python", "bin/range_http_server.py", "8001", "--bind", "0.0.0.0"]
