#!/usr/bin/env python3
"""
Predict an absorption spectrum from an electron-density cube.

The input is a molecular ground-state electron density, rho(r), stored as a
NumPy .npz volume. The script recentres and rescales the density in the same
spirit as the training pipeline, then passes it through a saved 3D CNN. The
output is a normalised absorption spectrum on a fixed energy grid from
0.0 to 0.45 atomic units. When a matching TDDFT reference spectrum is available,
it is written beside the prediction for direct comparison.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.ndimage import gaussian_filter1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# The density grid used during training: channel, x, y, z.
DENSITY_SHAPE = (1, 155, 147, 143)

# Energy range of the predicted absorption spectrum, in atomic units.
ENERGY_MIN_AU = 0.0
ENERGY_MAX_AU = 0.45


class CNNModel(nn.Module):
    """3D CNN definition needed when loading a model saved with torch.save(model)."""

    def __init__(
        self,
        num_layers: int = 8,
        kernel_size: int = 5,
        base_filters: int = 4,
        dropout_p: float = 0.1,
        input_shape: tuple[int, int, int, int] = DENSITY_SHAPE,
        out_features: int = 1,
    ) -> None:
        super().__init__()

        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.base_filters = int(base_filters)
        self.dropout_p = float(dropout_p)
        self.input_shape = tuple(input_shape)
        self.in_ch = int(input_shape[0])
        self.out_features = int(out_features)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

        padding = (self.kernel_size - 1) // 2

        def channel_multiplier(layer_index: int) -> int:
            if layer_index == 1:
                return 1
            if layer_index == 2:
                return 2
            if layer_index == 3:
                return 4
            if layer_index == 4:
                return 8
            if layer_index == 5:
                return 16
            return 32

        blocks = []
        in_channels = self.in_ch

        for layer_index in range(1, self.num_layers + 1):
            out_channels = self.base_filters * channel_multiplier(layer_index)
            out_channels = min(out_channels, self.base_filters * 32)

            blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=self.kernel_size,
                        padding=padding,
                        padding_mode="zeros",
                    ),
                    nn.GroupNorm(out_channels, out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.drop_mid = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.drop_end = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(self.out_features)
        self.softmax = nn.Softmax(dim=1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        dropout_after = max(1, math.floor(self.num_layers * 0.6))

        for layer_index, block in enumerate(self.blocks, start=1):
            x = block(x)
            x = self.avgpool(self.maxpool(x))

            if layer_index == dropout_after:
                x = self.drop_mid(x)

        return self.drop_end(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.softmax(x)


def load_density(npz_path: Path) -> np.ndarray:
    """Load the electron-density volume from a .npz file."""
    with np.load(npz_path, allow_pickle=False) as archive:
        if "data" not in archive.files:
            available = ", ".join(archive.files)
            raise KeyError(f"Expected key 'data' in {npz_path}. Available keys: {available}")
        density = archive["data"]

    if density.ndim == 3:
        volume = density
    elif density.ndim == 4:
        volume = density[..., 0]
    else:
        raise ValueError(f"Expected a 3D density volume, got shape {density.shape}")

    return volume.astype(np.float32, copy=False)


def centre_of_mass(volume: np.ndarray) -> np.ndarray:
    """Return the centre of mass in voxel-index coordinates."""
    weights = np.abs(volume)
    total_weight = float(weights.sum())
    nx, ny, nz = weights.shape

    if total_weight <= 0.0:
        return np.array(
            [(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0],
            dtype=np.float64,
        )

    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    z_axis = np.arange(nz, dtype=np.float64)

    x_centre = float((weights.sum(axis=(1, 2)) * x_axis).sum() / total_weight)
    y_centre = float((weights.sum(axis=(0, 2)) * y_axis).sum() / total_weight)
    z_centre = float((weights.sum(axis=(0, 1)) * z_axis).sum() / total_weight)

    return np.array([x_centre, y_centre, z_centre], dtype=np.float64)


def paste_with_aligned_centre(
    target: np.ndarray,
    source: np.ndarray,
    source_centre: np.ndarray,
) -> None:
    """Paste source into target after aligning source centre to target centre."""
    target_shape = np.array(target.shape)
    source_shape = np.array(source.shape)
    target_centre = (target_shape - 1) / 2.0

    start = np.round(target_centre - source_centre).astype(int)
    target_start = np.maximum(start, 0)
    target_stop = np.minimum(start + source_shape, target_shape)

    if np.any(target_stop <= target_start):
        return

    source_start = np.maximum(0, -start)
    source_stop = source_start + (target_stop - target_start)

    tx0, ty0, tz0 = target_start
    tx1, ty1, tz1 = target_stop
    sx0, sy0, sz0 = source_start
    sx1, sy1, sz1 = source_stop

    target[tx0:tx1, ty0:ty1, tz0:tz1] = source[sx0:sx1, sy0:sy1, sz0:sz1]


def prepare_density_tensor(npz_path: Path, density_shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Load, rescale, recentre, and batch the density for the CNN."""
    channels, nx, ny, nz = density_shape
    if channels != 1:
        raise ValueError("This inference script expects a single density channel.")

    volume = load_density(npz_path)

    max_abs_value = float(np.max(np.abs(volume)))
    if max_abs_value > 0.0:
        volume = volume / max_abs_value

    centred = np.zeros((nx, ny, nz), dtype=np.float32)
    paste_with_aligned_centre(centred, volume, source_centre=centre_of_mass(volume))

    # Shape expected by PyTorch Conv3d: batch, channel, x, y, z.
    return torch.from_numpy(centred[None, None, :, :, :]).float()


def infer_reference_path(npz_path: Path) -> Path | None:
    """Infer the conventional TDDFT spectrum path from density_M*.npz."""
    match = re.search(r"density_M(\d+)\.npz$", npz_path.name)
    if match is None:
        return None

    molecule_id = match.group(1)
    return npz_path.parent / f"tddft_spectrum_gamma_150meV_M{molecule_id}.dat"


def load_reference_file(ref_path: Path) -> tuple[np.ndarray | None, np.ndarray]:
    """Load a reference spectrum with either one column or energy/intensity columns."""
    data = np.loadtxt(ref_path)

    if data.ndim == 1:
        return None, data.astype(np.float32, copy=False)

    if data.shape[1] < 2:
        return None, data[:, 0].astype(np.float32, copy=False)

    energy = data[:, 0].astype(np.float32, copy=False)
    intensity = data[:, 1].astype(np.float32, copy=False)
    return energy, intensity


def prepare_reference_spectrum(
    ref_path: Path,
    energy_grid: np.ndarray,
    normalise: bool = True,
    smooth: bool = True,
    smoothing_width: float = 10.0,
) -> np.ndarray:
    """Put the reference spectrum on the prediction grid."""
    reference_energy, reference_intensity = load_reference_file(ref_path)
    reference_intensity = np.clip(reference_intensity.astype(np.float32), 0.0, None)

    if normalise and smooth and smoothing_width > 0:
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for reference smoothing. "
                "Install scipy or use --no-smooth-reference."
            )
        reference_intensity = gaussian_filter1d(
            reference_intensity,
            sigma=float(smoothing_width),
        ).astype(np.float32)

    if reference_energy is None:
        reference_energy = np.linspace(
            ENERGY_MIN_AU,
            ENERGY_MAX_AU,
            reference_intensity.size,
            dtype=np.float32,
        )

    reference_on_grid = np.interp(
        energy_grid,
        reference_energy,
        reference_intensity,
    ).astype(np.float32)

    if normalise:
        total = float(reference_on_grid.sum())
        if total > 0.0:
            reference_on_grid = reference_on_grid / total

    return reference_on_grid


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    """Load a full PyTorch model object and move it to the selected device."""
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)

    model.to(device)
    model.eval()
    return model


def predict_spectrum(model: nn.Module, density_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run inference and return a one-dimensional spectrum."""
    with torch.no_grad():
        prediction = model(density_tensor.to(device))

    prediction = prediction.detach().cpu().numpy()

    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise RuntimeError(f"Unexpected model output shape: {prediction.shape}")

    return prediction[0].astype(np.float32, copy=False)


def build_output_path(outdir: Path, npz_path: Path) -> Path:
    """Create a readable output filename from the density filename."""
    stem = npz_path.stem.replace("density_", "")
    return outdir / f"spectrum_prediction_{stem}.dat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a molecular absorption spectrum from an electron-density .npz file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the saved PyTorch CNN model.",
    )
    parser.add_argument(
        "--npz",
        required=True,
        type=Path,
        help="Path to the density_M*.npz input file.",
    )
    parser.add_argument(
        "--outdir",
        default=Path("predictions"),
        type=Path,
        help="Directory where the prediction file will be saved.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        type=Path,
        help="Optional reference spectrum. If omitted, the script searches next to the density file.",
    )
    parser.add_argument(
        "--raw-reference",
        action="store_true",
        help="Save the interpolated reference without training-style normalisation.",
    )
    parser.add_argument(
        "--no-smooth-reference",
        action="store_true",
        help="Do not smooth the reference spectrum before interpolation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for inference.",
    )

    return parser.parse_args()


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device)

    print(f"Preparing density: {args.npz}")
    density_tensor = prepare_density_tensor(args.npz, DENSITY_SHAPE)
    print(f"Input tensor shape: {tuple(density_tensor.shape)}")

    prediction = predict_spectrum(model, density_tensor, device)
    energy_grid = np.linspace(
        ENERGY_MIN_AU,
        ENERGY_MAX_AU,
        prediction.size,
        dtype=np.float32,
    )

    reference_path = args.ref if args.ref is not None else infer_reference_path(args.npz)
    has_reference = reference_path is not None and reference_path.exists()

    header_lines = [
        "energy_au prediction" if not has_reference else "energy_au reference prediction",
        f"model={args.model}",
        f"density_npz={args.npz}",
    ]

    if has_reference:
        print(f"Using reference spectrum: {reference_path}")
        reference = prepare_reference_spectrum(
            reference_path,
            energy_grid=energy_grid,
            normalise=not args.raw_reference,
            smooth=not args.no_smooth_reference,
        )
        output = np.column_stack([energy_grid, reference, prediction])
        header_lines.extend(
            [
                f"reference_dat={reference_path}",
                f"reference_mode={'raw_interpolated' if args.raw_reference else 'training_scaled'}",
            ]
        )
    else:
        print("No reference spectrum found. Saving prediction only.")
        output = np.column_stack([energy_grid, prediction])

    output_path = build_output_path(args.outdir, args.npz)
    np.savetxt(
        output_path,
        output,
        fmt="%.10e",
        header="\n".join(header_lines),
        comments="# ",
    )

    print(f"Saved prediction to: {output_path}")
    print(f"Prediction points: {prediction.size}")
    print(f"Prediction sum: {float(prediction.sum()):.6f}")

    if has_reference:
        print(f"Reference sum: {float(reference.sum()):.6f}")


if __name__ == "__main__":
    main()
