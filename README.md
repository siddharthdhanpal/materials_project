# Spectrum training and prediction scripts

This README describes the data layout, environment setup, training scripts, and prediction scripts used for molecular absorption-spectrum prediction. It is intended for the files currently organised as:

```text
materials_project/
  data/
    GEO_M1.xyz
    density_M1.npz
    tddft_spectrum_gamma_150meV_M1.dat

  scripts/
    predict_spectrum_from_density.py
    predict_spectrum_from_xyz_schnet.py
    predict_mace_spectrum.py

  trainingScripts/
    build_density_spectrum_cache.py
    sweep_graph_spectrum_models_from_cache.py
    sweep_schnet_dimenetpp_spectrum_from_cache.py
    train_mace_spectrum_from_cache.py
    train_mace_spectrum_sweep_from_cache.py
```

The `data/` directory shown here is a small example layout for testing prediction commands. The full training dataset, generated caches, run outputs, and model weights are usually too large for normal GitHub commits and should be stored outside the repository or provided separately.

## 1. Environment

Create the conda environment before training or prediction:

```bash
conda env create -f environment.yml
conda activate <environment-name-from-environment-yml>
```

Training requires the full ML environment, including PyTorch, PyTorch Geometric, MACE/e3nn, NumPy, SciPy, and plotting dependencies. Prediction also needs the corresponding model dependencies: the density model uses PyTorch and SciPy, the SchNet/DimeNet++ prediction script uses PyTorch Geometric, and the MACE prediction script uses MACE/e3nn.

## 2. Data files and naming convention

The scripts connect density, geometry, and spectrum files through the molecule id in the filename. For molecule `1`, the expected filenames are:

```text
GEO_M1.xyz
 density_M1.npz
 tddft_spectrum_gamma_150meV_M1.dat
```

The example repository layout is flat:

```text
data/
  GEO_M1.xyz
  density_M1.npz
  tddft_spectrum_gamma_150meV_M1.dat
```

For training on the full dataset, the recommended layout is directory-per-molecule:

```text
/path/to/density-spectrum-root/
  1/
    density_M1.npz
    tddft_spectrum_gamma_150meV_M1.dat
  2/
    density_M2.npz
    tddft_spectrum_gamma_150meV_M2.dat
  ...

/path/to/xyz-root/
  1/
    GEO_M1.xyz
  2/
    GEO_M2.xyz
  ...
```

Different directory structures are supported through command-line templates. Do not hard-code local machine paths in the scripts.

## 3. Cache-first training workflow

Run the cache builder first. It creates memory-mappable density and spectrum shards and writes an `index.json`. The same `index.json` should then be passed to graph and MACE training scripts so that molecule ordering and the train/validation split remain consistent across model families.

### 3.1 Build the cache only

For the recommended directory-per-molecule layout:

```bash
python trainingScripts/build_density_spectrum_cache.py \
  --root /path/to/density-spectrum-root \
  --cache-dir /path/to/cache-dir \
  --prepare-only
```

For the flat example-style layout:

```bash
python trainingScripts/build_density_spectrum_cache.py \
  --root /path/to/data \
  --density-glob "{root}/density_M*.npz" \
  --spectrum-template "{density_dir}/tddft_spectrum_gamma_150meV_M{mid}.dat" \
  --cache-dir /path/to/cache-dir \
  --prepare-only
```

The important output is:

```text
/path/to/cache-dir/index.json
```

The cache directory also contains files such as `cubes_shard_0000.npy` and `specs_shard_0000.npy`. Keep these files together with `index.json`. If the dataset is moved later, rebuild the cache or use the path-template options in the downstream scripts.

### 3.2 Optional CNN density-to-spectrum training

The same cache-building script can also train the 3D CNN density-to-spectrum model:

```bash
python trainingScripts/build_density_spectrum_cache.py \
  --root /path/to/density-spectrum-root \
  --cache-dir /path/to/cache-dir \
  --outdir /path/to/cnn-runs \
  --experiments 20 \
  --epochs 200
```

Use `--prepare-only` when you only want to build the shared cache and not start CNN training.

## 4. Graph and MACE training scripts

All scripts below consume the cache index produced in the previous step.

### 4.1 SchNet and DimeNet++ sweep

```bash
python trainingScripts/sweep_schnet_dimenetpp_spectrum_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --outdir /path/to/graph-runs \
  --model both \
  --runs 10 \
  --epochs 100
```

For a flat XYZ layout, use:

```bash
python trainingScripts/sweep_schnet_dimenetpp_spectrum_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/data \
  --xyz-template "{xyz_root}/GEO_M{mid}.xyz" \
  --outdir /path/to/graph-runs \
  --model both
```

### 4.2 Four-model graph sweep

This compares SchNet, SchNet with attention-style pooling, DimeNet++, and a DimeNet++ graph-head variant:

```bash
python trainingScripts/sweep_graph_spectrum_models_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --outdir /path/to/four-model-runs \
  --runs-per-model 5 \
  --epochs 200
```

### 4.3 Single MACE training run or size-scaling run

```bash
python trainingScripts/train_mace_spectrum_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --outdir /path/to/mace-runs \
  --epochs 200
```

You can train on fractions of the training set with:

```bash
python trainingScripts/train_mace_spectrum_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --outdir /path/to/mace-size-runs \
  --fractions 0.25 0.50 1.0 \
  --epochs 200
```

### 4.4 MACE hyperparameter sweep

This script was written to reuse a cache produced by the current cache builder or a previously produced compatible cache. It can read cached spectrum shards when available.

```bash
python trainingScripts/train_mace_spectrum_sweep_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --outdir /path/to/mace-spectrum-sweep \
  --target-source cache \
  --epochs 400
```

Use raw spectrum files recorded in the cache index instead of cached spectrum shards with:

```bash
python trainingScripts/train_mace_spectrum_sweep_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --outdir /path/to/mace-spectrum-sweep \
  --target-source raw
```

For a different XYZ or spectrum layout:

```bash
python trainingScripts/train_mace_spectrum_sweep_from_cache.py \
  --cache-index /path/to/cache-dir/index.json \
  --xyz-root /path/to/xyz-root \
  --xyz-template "{xyz_root}/molecules/{mid}/geometry.xyz" \
  --spectrum-template "/path/to/spectra/{mid}/spectrum.dat" \
  --outdir /path/to/mace-spectrum-sweep
```

## 5. Prediction scripts

Model weights are not included at this stage. When trained weights or best-model files are provided later, pass their paths using the commands below.

### 5.1 Predict from an electron-density `.npz` file

Use this for a 3D CNN density-to-spectrum model:

```bash
python scripts/predict_spectrum_from_density.py \
  --model /path/to/cnn_model.pt \
  --npz data/density_M1.npz \
  --outdir predictions/density_M1
```

With an explicit TDDFT reference:

```bash
python scripts/predict_spectrum_from_density.py \
  --model /path/to/cnn_model.pt \
  --npz data/density_M1.npz \
  --ref data/tddft_spectrum_gamma_150meV_M1.dat \
  --outdir predictions/density_M1
```

The output is a text table with the energy grid and prediction. If a reference is supplied, or if a matching reference is found next to the density file, the reference is included for comparison.

### 5.2 Predict from XYZ using SchNet or DimeNet++

Use this for graph models saved as full `.pt` models or portable checkpoint bundles:

```bash
python scripts/predict_spectrum_from_xyz_schnet.py \
  --model /path/to/graph_model.pt \
  --xyz data/GEO_M1.xyz \
  --reference data/tddft_spectrum_gamma_150meV_M1.dat \
  --outdir predictions/graph_M1
```

The reference is optional:

```bash
python scripts/predict_spectrum_from_xyz_schnet.py \
  --model /path/to/graph_model.pt \
  --xyz data/GEO_M1.xyz \
  --outdir predictions/graph_M1
```

### 5.3 Predict from XYZ using MACE

Use this for a MACE `best_model.pth` state dictionary:

```bash
python scripts/predict_mace_spectrum.py \
  --weights /path/to/best_model.pth \
  --xyz data/GEO_M1.xyz \
  --reference data/tddft_spectrum_gamma_150meV_M1.dat \
  --outdir predictions/mace_M1
```

The reference is optional:

```bash
python scripts/predict_mace_spectrum.py \
  --weights /path/to/best_model.pth \
  --xyz data/GEO_M1.xyz \
  --outdir predictions/mace_M1
```

The MACE prediction directory contains a text table, NumPy arrays, metadata, and a quick plot when plotting dependencies are installed.

## 6. Outputs to expect

Training scripts usually write:

```text
<run-output>/
  summary.csv
  summary.json                 # when available
  split.json                   # when available
  <experiment>/
    config.json
    train_log.csv
    best_model.pth
    val_preds_true.npz
```

Prediction scripts usually write:

```text
predictions/<run-name>/
  *.dat                        # energy grid and prediction/reference columns
  *.npz                        # NumPy arrays, when supported by the script
  *.json                       # metadata, when supported by the script
  *.png                        # quick plot, when supported by the script
```

## 7. Important conventions

- The default target is a fixed-length normalized absorption spectrum on an energy/frequency grid from `0.0` to `0.45` atomic units.
- Spectrum intensities are clipped at zero when required and normalized so that the target sums to one.
- The cache index is the source of molecule ordering for downstream scripts.
- The default split is deterministic, controlled by `--seed` and `--val-frac`.
- The full training data, generated caches, and model weights should not be committed unless they are intentionally managed through Git LFS, releases, or an external data store.

## 8. Common issues

### Cache paths no longer exist

If `index.json` points to old local paths, rebuild the cache in the new location. For graph and MACE scripts, you can also pass `--spectrum-template` and `--xyz-template` to reconstruct paths without editing the code.

### XYZ files are in a flat directory

Pass:

```bash
--xyz-root /path/to/data \
--xyz-template "{xyz_root}/GEO_M{mid}.xyz"
```

### Density and spectrum files are in a flat directory

Pass:

```bash
--density-glob "{root}/density_M*.npz" \
--spectrum-template "{density_dir}/tddft_spectrum_gamma_150meV_M{mid}.dat"
```

### Prediction fails because model weights are missing

Prediction scripts need trained weights. These are expected to be provided separately in the future. Once available, pass them through `--model` for CNN/SchNet/DimeNet++ prediction or `--weights` for MACE prediction.
