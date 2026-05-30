## Import Dependencies


```python
import random
import re
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## Random State & Device Initialization


```python
SEED = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
```


```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")
```

## Formatting


```python
PLOT_DPI = 165

COLORS = {
    "train": "#2563EB",
    "valid": "#F97316",
    "test": "#16A34A",
    "adam": "#2563EB",
    "lbfgs": "#9333EA",
    "data": "#0EA5E9",
    "physics": "#DC2626",
    "reflection": "#16A34A",
    "ideal": "#111827",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "purple": "#7C3AED",
    "amber": "#F59E0B",
}

SPLIT_COLORS = {
    "train": COLORS["train"],
    "valid": COLORS["valid"],
    "test": COLORS["test"],
}

plt.rcParams.update({
    "figure.figsize": (7.2, 4.4),
    "figure.dpi": PLOT_DPI,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.labelcolor": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.8,
    "grid.color": "#9CA3AF",
    "lines.linewidth": 2.1,
    "legend.frameon": True,
    "legend.framealpha": 0.94,
    "legend.edgecolor": "#E5E7EB",
})


def polish_axes(
    ax,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
) -> None:
    if title is not None:
        ax.set_title(title, pad=11)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=7)

    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_alpha(0.7)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#6B7280")

    if legend:
        leg = ax.legend()
        if leg is not None:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#E5E7EB")
            leg.get_frame().set_linewidth(0.9)
            leg.get_frame().set_alpha(0.95)


def smooth_series(values, window: int = 35) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def log10_safe(values, eps: float = 1e-16) -> np.ndarray:
    return np.log10(np.asarray(values).clip(min=eps))


def plot_density_hist(
    ax,
    values,
    *,
    bins: int = 80,
    color: str,
    label: str,
    alpha: float = 0.20,
    linewidth: float = 2.2,
) -> None:
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=alpha,
        color=color,
        edgecolor=color,
        linewidth=1.0,
        label=label,
    )
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="step",
        color=color,
        linewidth=linewidth,
    )


def plot_ecdf(
    ax,
    values,
    *,
    color: str,
    label: str,
    linewidth: float = 2.2,
) -> None:
    x = np.sort(np.asarray(values))
    y = np.linspace(0.0, 1.0, len(x), endpoint=True)
    ax.plot(x, y, color=color, linewidth=linewidth, label=label)


def compact_count(n: int) -> str:
    n = int(n)
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def compact_float(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m").replace("+", "")


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


def save_dataframe_as_png(df_to_save: pd.DataFrame, path: Path, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    display_df = df_to_save.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.6g}" if pd.notna(x) else "")

    fig_height = max(1.8, 0.42 * (len(display_df) + 1) + (0.35 if title else 0.0))
    fig_width = max(8.0, 1.35 * len(display_df.columns))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=12.5, fontweight="semibold", pad=10)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_text_props(weight="semibold", color="#111827")
            cell.set_facecolor("#F3F4F6")
        else:
            cell.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dataframe_text(df_to_save: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df_to_save.to_string(index=False), encoding="utf-8")
```

## Configuration


```python
@dataclass(frozen=True)
class Config:
    notebook_dir: Path = Path.cwd()
    project_root: Path = notebook_dir.parent if notebook_dir.name.lower() == "pinn" else notebook_dir

    data_path: Path = project_root / "storage" / "dataset_10k.csv"
    saved_models_dir: Path = (
        notebook_dir / "saved_models"
        if notebook_dir.name.lower() == "pinn"
        else project_root / "PINN" / "saved_models"
    )
    visualization_dir: Path = project_root / "visualization"

    load_existing_model: bool = False
    model_checkpoint_path: Path | None = None
    save_model_after_training: bool = True
    model_run_id: int | None = None

    plate_rho: float = 7800.0
    plate_h: float = 0.002
    plate_E: float = 2.0e11
    plate_nu: float = 0.3

    omega0: float = 2.0 * np.pi * 100.0
    period_a: float = 0.05

    min_R: float = 0.0
    max_R: float = 1.0

    add_noise: bool = False
    noise_level: float = 0.0
    noise_seed: int = 42

    test_size: float = 0.15
    valid_size: float = 0.15
    n_strat_bins: int = 10

    batch_size: int = 256
    use_weighted_sampler: bool = True
    sampler_high_R_scale: float = 1.0
    max_epochs: int = 2000
    learning_rate: float = 5.0e-4
    weight_decay: float = 5.0e-4

    use_lbfgs: bool = True
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 5000
    lbfgs_max_eval: int = 7500
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1.0e-9
    lbfgs_tolerance_change: float = 1.0e-11
    lbfgs_log_every: int = 10
    lbfgs_valid_log_every: int = 25

    lambda_phys: float = 0.5
    lambda_R: float = 2.0
    R_loss_eps: float = 1.0e-6
    R_loss_log_weight: float = 0.85
    R_loss_sqrt_weight: float = 0.15

    patience: int = 180
    min_delta: float = 1.0e-7

    hidden_dim: int = 160
    num_hidden_layers: int = 4
    dropout: float = 0.02

    grad_clip_norm: float = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 60

    numerical_eps: float = 1.0e-12
    spectrum_n_terms: int = 80
    resonator_damping_ratio: float = 0.01
    resonator_mu_scale: float = 1.0

    fixed_spectrum_points: int = 500
    reference_psi: float = np.pi / 6.0
    near_zero_delta_y: float = 1.0e-4

    geometry_grid_size: int = 72
    geometry_kappa_quantile_low: float = 0.85
    geometry_kappa_quantile_high: float = 0.95
    zeta_error_eps: float = 1.0e-12

    plot_sample_size: int = 10_000
    relative_error_min_R: float = 1.0e-4

    use_spectrum_augmentation: bool = True
    spectrum_points_per_target_case: int = 800
    spectrum_random_cases: int = 6
    spectrum_points_per_random_case: int = 180
    spectrum_augmentation_seed: int = 2026


cfg = Config()

print("Current directory:", Path.cwd())
print("Dataset path:", cfg.data_path)
print("Saved models root:", cfg.saved_models_dir)
print("Visualization root:", cfg.visualization_dir)
print(cfg)
```

## Spectrum Augmentation


```python
@dataclass(frozen=True)
class AugPlateParams:
    rho: float
    h: float
    E: float
    nu: float

    @property
    def D(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))


@dataclass(frozen=True)
class AugResonatorParams:
    omega0: float
    damping_ratio: float
    mu_scale: float

    @property
    def gamma(self) -> float:
        return self.damping_ratio * self.omega0


def aug_omega_from_kappa(kappa_value: np.ndarray, plate: AugPlateParams) -> np.ndarray:
    return np.sqrt((np.asarray(kappa_value) ** 4) * plate.D / (plate.rho * plate.h))


def aug_mu_resonator(omega: np.ndarray, resonator: AugResonatorParams) -> np.ndarray:
    denominator = resonator.omega0**2 - omega**2 - 1j * resonator.gamma * omega
    return -resonator.mu_scale * omega**2 / denominator


def aug_lambda_gamma(kxn: np.ndarray, kappa_value: float) -> tuple[np.ndarray, np.ndarray]:
    ratio = kxn / kappa_value
    ratio2 = ratio * ratio

    lam = np.empty_like(ratio, dtype=np.complex128)
    propagating = np.abs(ratio) < 1.0

    lam[propagating] = -1j * np.sqrt(1.0 - ratio2[propagating])
    lam[~propagating] = np.sqrt(ratio2[~propagating] - 1.0) + 0.0j

    gam = np.sqrt(ratio2 + 1.0).astype(np.complex128)
    return lam, gam


def aug_lattice_sums(
    psi: float,
    delta_x: float,
    delta_y: float,
    kappa_value: float,
    period_a: float,
    n_terms: int = 80,
    eps: float = 1.0e-12,
) -> tuple[complex, complex, complex]:
    n_values = np.arange(-n_terms, n_terms + 1, dtype=np.float64)
    kx = kappa_value * np.cos(psi)
    kxn = kx + 2.0 * np.pi * n_values / period_a

    lam, gam = aug_lambda_gamma(kxn, kappa_value)
    valid = (np.abs(lam) >= eps) & (np.abs(gam) >= eps)

    inv_term = np.zeros_like(lam, dtype=np.complex128)
    inv_term[valid] = 1.0 / lam[valid] - 1.0 / gam[valid]

    norm = 4.0 * period_a * kappa_value**3
    S = np.sum(inv_term) / norm

    dy = abs(delta_y)
    cross_term = np.zeros_like(lam, dtype=np.complex128)
    cross_term[valid] = (
        np.exp(-kappa_value * lam[valid] * dy) / lam[valid]
        - np.exp(-kappa_value * gam[valid] * dy) / gam[valid]
    )

    phase = np.exp(1j * kx * delta_x)
    S1 = np.conj(phase) * np.sum(cross_term) / norm
    S2 = phase * np.sum(cross_term) / norm

    return complex(S), complex(S1), complex(S2)


def aug_solve_zeta(
    psi: float,
    delta_y: float,
    mu: complex,
    S: complex,
    S1: complex,
    S2: complex,
    kappa_value: float,
) -> tuple[complex, complex, complex]:
    ky = kappa_value * np.sin(psi)

    exp1 = 1.0 + 0.0j
    exp2 = np.exp(1j * ky * delta_y)

    denominator = (1.0 - mu * S) ** 2 - mu**2 * S1 * S2
    zeta1 = ((1.0 - mu * S) * exp1 + mu * S1 * exp2) / denominator
    zeta2 = ((1.0 - mu * S) * exp2 + mu * S2 * exp1) / denominator

    return zeta1, zeta2, denominator


def aug_reflection_transmission(
    psi: float,
    delta_y: float,
    mu: complex,
    zeta1: complex,
    zeta2: complex,
    kappa_value: float,
    period_a: float,
) -> tuple[float, float, complex]:
    ky = kappa_value * np.sin(psi)
    amplitude = zeta1 + zeta2 * np.exp(1j * ky * delta_y)
    R0 = 1j * mu * amplitude / (4.0 * period_a * kappa_value**3 * np.sin(psi))
    R = float(abs(R0) ** 2)
    T = float(abs(1.0 + R0) ** 2)
    return R, T, R0


def build_spectrum_rows_for_case(
    psi: float,
    delta_x: float,
    delta_y: float,
    kappa_grid: np.ndarray,
    cfg: Config,
) -> list[dict[str, float]]:
    plate = AugPlateParams(
        rho=cfg.plate_rho,
        h=cfg.plate_h,
        E=cfg.plate_E,
        nu=cfg.plate_nu,
    )
    resonator = AugResonatorParams(
        omega0=cfg.omega0,
        damping_ratio=cfg.resonator_damping_ratio,
        mu_scale=cfg.resonator_mu_scale,
    )

    omega_grid = aug_omega_from_kappa(kappa_grid, plate)
    mu_grid = aug_mu_resonator(omega_grid, resonator)

    rows = []
    for kap, omega, mu in zip(kappa_grid, omega_grid, mu_grid):
        try:
            S, S1, S2 = aug_lattice_sums(
                psi=psi,
                delta_x=delta_x,
                delta_y=delta_y,
                kappa_value=float(kap),
                period_a=cfg.period_a,
                n_terms=cfg.spectrum_n_terms,
                eps=cfg.numerical_eps,
            )
            zeta1, zeta2, denominator = aug_solve_zeta(
                psi=psi,
                delta_y=delta_y,
                mu=complex(mu),
                S=S,
                S1=S1,
                S2=S2,
                kappa_value=float(kap),
            )
            R, T, R0 = aug_reflection_transmission(
                psi=psi,
                delta_y=delta_y,
                mu=complex(mu),
                zeta1=zeta1,
                zeta2=zeta2,
                kappa_value=float(kap),
                period_a=cfg.period_a,
            )
        except FloatingPointError:
            continue

        row = {
            "omega": float(omega),
            "psi": float(psi),
            "delta_x": float(delta_x),
            "delta_y": float(delta_y),
            "kappa": float(kap),
            "Re_mu": float(np.real(mu)),
            "Im_mu": float(np.imag(mu)),
            "Re_S": float(np.real(S)),
            "Im_S": float(np.imag(S)),
            "Re_S1": float(np.real(S1)),
            "Im_S1": float(np.imag(S1)),
            "Re_S2": float(np.real(S2)),
            "Im_S2": float(np.imag(S2)),
            "Re_zeta1": float(np.real(zeta1)),
            "Im_zeta1": float(np.imag(zeta1)),
            "Re_zeta2": float(np.real(zeta2)),
            "Im_zeta2": float(np.imag(zeta2)),
            "R": float(np.clip(R, cfg.min_R, cfg.max_R)),
            "T": float(T),
            "Re_R0": float(np.real(R0)),
            "Im_R0": float(np.imag(R0)),
            "energy_balance": float(R + T),
            "abs_denominator": float(abs(denominator)),
            "residual_1": 0.0,
            "residual_2": 0.0,
        }

        if np.all(np.isfinite(list(row.values()))) and cfg.min_R <= row["R"] <= cfg.max_R:
            rows.append(row)

    return rows


def make_kappa_grid_for_spectra(kappa_min: float, kappa_max: float, n_points: int, cfg: Config) -> np.ndarray:
    plate = AugPlateParams(
        rho=cfg.plate_rho,
        h=cfg.plate_h,
        E=cfg.plate_E,
        nu=cfg.plate_nu,
    )
    kappa0 = float((plate.rho * plate.h * cfg.omega0**2 / plate.D) ** 0.25)

    n_uniform = max(50, int(0.65 * n_points))
    n_resonance = max(20, n_points - n_uniform)

    uniform_grid = np.linspace(kappa_min, kappa_max, n_uniform)

    resonance_width = max(0.5, 0.12 * (kappa_max - kappa_min))
    resonance_grid = np.linspace(
        max(kappa_min, kappa0 - resonance_width),
        min(kappa_max, kappa0 + resonance_width),
        n_resonance,
    )

    grid = np.unique(np.concatenate([uniform_grid, resonance_grid]))
    return grid


def build_spectrum_augmentation_df(base_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    kappa_min = max(float(base_df["kappa"].min()), 1.0e-3)
    kappa_max = float(base_df["kappa"].max())
    rng = np.random.default_rng(cfg.spectrum_augmentation_seed)

    rows: list[dict[str, float]] = []

    target_cases = [
        {
            "psi": cfg.reference_psi,
            "delta_x": 0.5 * cfg.period_a,
            "delta_y": cfg.near_zero_delta_y,
        },
        {
            "psi": cfg.reference_psi,
            "delta_x": 0.5 * cfg.period_a,
            "delta_y": 0.5 * cfg.period_a * np.tan(np.pi / 6.0),
        },
    ]

    target_grid = make_kappa_grid_for_spectra(
        kappa_min,
        kappa_max,
        cfg.spectrum_points_per_target_case,
        cfg,
    )

    for case in target_cases:
        rows.extend(build_spectrum_rows_for_case(
            psi=case["psi"],
            delta_x=case["delta_x"],
            delta_y=case["delta_y"],
            kappa_grid=target_grid,
            cfg=cfg,
        ))

    random_grid = make_kappa_grid_for_spectra(
        kappa_min,
        kappa_max,
        cfg.spectrum_points_per_random_case,
        cfg,
    )

    for _ in range(cfg.spectrum_random_cases):
        psi = float(rng.uniform(cfg.near_zero_delta_y, np.pi - cfg.near_zero_delta_y))
        delta_x = float(rng.uniform(0.0, cfg.period_a))
        delta_y = float(rng.uniform(cfg.near_zero_delta_y, cfg.period_a))

        rows.extend(build_spectrum_rows_for_case(
            psi=psi,
            delta_x=delta_x,
            delta_y=delta_y,
            kappa_grid=random_grid,
            cfg=cfg,
        ))

    aug_df = pd.DataFrame(rows)
    if len(aug_df) == 0:
        raise RuntimeError("No rows produced")

    return aug_df
```

## Data Loading


```python
df = pd.read_csv(cfg.data_path)

required_columns = [
    "omega", "psi", "delta_x", "delta_y", "kappa",
    "Re_mu", "Im_mu",
    "Re_S", "Im_S", "Re_S1", "Im_S1", "Re_S2", "Im_S2",
    "Re_zeta1", "Im_zeta1", "Re_zeta2", "Im_zeta2",
    "R", "T",
]

missing = sorted(set(required_columns) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

finite_mask = np.ones(len(df), dtype=bool)
for col in required_columns:
    finite_mask &= np.isfinite(df[col].to_numpy())

if not finite_mask.all():
    bad_count = int((~finite_mask).sum())
    raise ValueError(f"Dataset contains {bad_count} rows with non-finite values")

R_mask = (df["R"] >= cfg.min_R) & (df["R"] <= cfg.max_R)
if not R_mask.all():
    bad_count = int((~R_mask).sum())
    raise ValueError(f"Dataset contains {bad_count} rows outside [{cfg.min_R}, {cfg.max_R}]")

base_n_objects = len(df)
base_n_objects_tag = compact_count(base_n_objects)

noise_percent = int(round((cfg.noise_level if cfg.add_noise else 0.0) * 100))
model_dir = cfg.saved_models_dir / f"noise_{noise_percent}"
model_dir.mkdir(parents=True, exist_ok=True)

visualization_parent_dir = cfg.visualization_dir / f"noise_{noise_percent}" / base_n_objects_tag
visualization_parent_dir.mkdir(parents=True, exist_ok=True)

aug_tag = "aug" if cfg.use_spectrum_augmentation else "noaug"
sampler_tag = f"ws{compact_float(cfg.sampler_high_R_scale)}" if cfg.use_weighted_sampler else "plain"

base_model_name = (
    f"pinn_{aug_tag}_{sampler_tag}_N{base_n_objects_tag}_"
    f"h{cfg.hidden_dim}x{cfg.num_hidden_layers}_"
    f"bs{cfg.batch_size}_ep{cfg.max_epochs}_"
    f"lr{compact_float(cfg.learning_rate)}_"
    f"phys{compact_float(cfg.lambda_phys)}_refl{compact_float(cfg.lambda_R)}"
)


def extract_run_id_from_checkpoint(path: Path) -> int | None:
    match = re.search(r"_m(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def existing_model_run_ids(model_dir: Path, base_name: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(base_name)}_m(\d+)\.pt$")
    run_ids = []
    for path in model_dir.glob(f"{base_name}_m*.pt"):
        match = pattern.match(path.name)
        if match:
            run_ids.append(int(match.group(1)))
    return run_ids


def existing_visualization_run_ids(parent_dir: Path) -> list[int]:
    return [
        int(path.name)
        for path in parent_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]


def resolve_run_id() -> int:
    if cfg.load_existing_model:
        if cfg.model_checkpoint_path is None:
            raise ValueError("Set cfg.model_checkpoint_path when cfg.load_existing_model=True")

        checkpoint_run_id = extract_run_id_from_checkpoint(Path(cfg.model_checkpoint_path))
        if cfg.model_run_id is not None:
            return cfg.model_run_id
        if checkpoint_run_id is not None:
            return checkpoint_run_id
        return 1

    if cfg.model_run_id is not None:
        return cfg.model_run_id

    used_ids = (
        existing_model_run_ids(model_dir, base_model_name)
        + existing_visualization_run_ids(visualization_parent_dir)
    )
    return max(used_ids, default=0) + 1


resolved_model_run_id = resolve_run_id()
model_path = model_dir / f"{base_model_name}_m{resolved_model_run_id:02d}.pt"
checkpoint_path = Path(cfg.model_checkpoint_path) if cfg.model_checkpoint_path is not None else model_path

visualization_run_dir = visualization_parent_dir / f"{resolved_model_run_id:02d}"
visualization_run_dir.mkdir(parents=True, exist_ok=True)

print(f"Loaded base rows: {base_n_objects}")
print(f"Columns: {len(df.columns)}")
print(f"Model path: {model_path}")
print(f"Checkpoint path: {checkpoint_path}")
print(f"Visualization path: {visualization_run_dir}")

display(df.head())
```

## Data Distribution


```python
R_values = df["R"].to_numpy()
log_R_values = log10_safe(R_values)

fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6), dpi=150)
fig.suptitle("Data Distribution", fontsize=16, fontweight="semibold", y=1.02)

axes[0, 0].hist(
    R_values,
    bins=90,
    color=COLORS["train"],
    alpha=0.88,
    edgecolor="white",
    linewidth=0.4,
)
polish_axes(
    axes[0, 0],
    "Distribution of $R$",
    "$R$",
    "Number of observations",
)

plot_density_hist(
    axes[0, 1],
    log_R_values,
    bins=90,
    color=COLORS["purple"],
    label=r"$\log_{10}(R)$",
    alpha=0.24,
)
polish_axes(
    axes[0, 1],
    r"Distribution of $\log_{10}(R)$",
    r"$\log_{10}(R)$",
    "Density",
    legend=True,
)

plot_ecdf(
    axes[1, 0],
    log_R_values,
    color=COLORS["lbfgs"],
    label="ECDF",
)
polish_axes(
    axes[1, 0],
    "ECDF of $R$",
    r"$\log_{10}(R)$",
    "Share of observations",
    legend=True,
)

quantile_levels = np.array([0.50, 0.90, 0.95, 0.99])
quantile_values = np.quantile(R_values, quantile_levels)

axes[1, 1].bar(
    [f"q{int(q * 100)}" for q in quantile_levels],
    quantile_values,
    color=[COLORS["train"], COLORS["valid"], COLORS["test"], COLORS["amber"]],
    alpha=0.88,
    edgecolor="white",
    linewidth=0.7,
)
axes[1, 1].set_yscale("log")
for i, val in enumerate(quantile_values):
    axes[1, 1].text(
        i,
        val * 1.08,
        f"{val:.2e}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#111827",
    )
polish_axes(
    axes[1, 1],
    "$R$ Quantiles",
    "Quantile",
    "$R$ value",
)

plt.tight_layout()
plt.show()
```

## Features, Target, Physical Parameters


```python
def build_features(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)

    features["kappa"] = data["kappa"]

    features["sin_psi"] = np.sin(data["psi"])
    features["cos_psi"] = np.cos(data["psi"])

    phase_x = 2.0 * np.pi * data["delta_x"] / cfg.period_a
    features["sin_dx"] = np.sin(phase_x)
    features["cos_dx"] = np.cos(phase_x)

    features["dy_norm"] = data["delta_y"] / cfg.period_a

    features["Re_mu"] = data["Re_mu"]
    features["Im_mu"] = data["Im_mu"]

    features["Re_S"] = data["Re_S"]
    features["Im_S"] = data["Im_S"]
    features["Re_S1"] = data["Re_S1"]
    features["Im_S1"] = data["Im_S1"]
    features["Re_S2"] = data["Re_S2"]
    features["Im_S2"] = data["Im_S2"]

    return features


feature_cols = [
    "kappa",
    "sin_psi", "cos_psi",
    "sin_dx", "cos_dx",
    "dy_norm",
    "Re_mu", "Im_mu",
    "Re_S", "Im_S",
    "Re_S1", "Im_S1",
    "Re_S2", "Im_S2",
]

target_cols = ["Re_zeta1", "Im_zeta1", "Re_zeta2", "Im_zeta2"]

X_raw = build_features(df, cfg)[feature_cols]
y_raw = df[target_cols]

phys_cols = [
    "psi", "delta_y", "kappa",
    "Re_mu", "Im_mu",
    "Re_S", "Im_S", "Re_S1", "Im_S1", "Re_S2", "Im_S2",
    "R",
]
phys_raw = df[phys_cols]

print("Model input columns:", feature_cols)
print("X:", X_raw.shape)
print("y:", y_raw.shape)
print("Physics:", phys_raw.shape)
```

## Train / Validation / Test Split


```python
df_split = df.copy()
df_split["log_R"] = np.log1p(df_split["R"])

df_split["R_bin"] = pd.qcut(
    df_split["log_R"],
    q=cfg.n_strat_bins,
    labels=False,
    duplicates="drop",
)

idx = np.arange(len(df_split))
strat_labels = df_split["R_bin"].astype(int).values

idx_train_base, idx_temp = train_test_split(
    idx,
    test_size=cfg.valid_size + cfg.test_size,
    random_state=SEED,
    stratify=strat_labels,
)

temp_test_fraction = cfg.test_size / (cfg.valid_size + cfg.test_size)

idx_valid, idx_test = train_test_split(
    idx_temp,
    test_size=temp_test_fraction,
    random_state=SEED,
    stratify=strat_labels[idx_temp],
)

X_train_raw = X_raw.iloc[idx_train_base].copy()
X_valid_raw = X_raw.iloc[idx_valid].copy()
X_test_raw = X_raw.iloc[idx_test].copy()

y_train_raw = y_raw.iloc[idx_train_base].copy()
y_valid_raw = y_raw.iloc[idx_valid].copy()
y_test_raw = y_raw.iloc[idx_test].copy()

phys_train_raw = phys_raw.iloc[idx_train_base].copy()
phys_valid_raw = phys_raw.iloc[idx_valid].copy()
phys_test_raw = phys_raw.iloc[idx_test].copy()
```


```python
n_train_base = len(X_train_raw)
n_aug = 0

if cfg.use_spectrum_augmentation:
    spectrum_aug_df = build_spectrum_augmentation_df(df, cfg)
    n_aug = len(spectrum_aug_df)

    X_aug_raw = build_features(spectrum_aug_df, cfg)[feature_cols]
    y_aug_raw = spectrum_aug_df[target_cols]
    phys_aug_raw = spectrum_aug_df[phys_cols]

    X_train_raw = pd.concat([X_train_raw, X_aug_raw], ignore_index=True)
    y_train_raw = pd.concat([y_train_raw, y_aug_raw], ignore_index=True)
    phys_train_raw = pd.concat([phys_train_raw, phys_aug_raw], ignore_index=True)

    print(
        "Train-only spectrum augmentation: "
        f"+{n_aug} rows ({n_train_base} -> {len(X_train_raw)})"
    )

split_parts = {
    "train": phys_train_raw,
    "valid": phys_valid_raw,
    "test": phys_test_raw,
}

split_labels = {
    "train": "Train + Spectrum Augmentation" if n_aug > 0 else "Train",
    "valid": "Validation",
    "test": "Test",
}

fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), dpi=150)
fig.suptitle("Distribution of $R$ by Split", fontsize=15.5, fontweight="semibold", y=1.04)

for name, part in split_parts.items():
    log_r = log10_safe(part["R"].to_numpy())
    color = SPLIT_COLORS[name]
    label = split_labels[name]

    plot_density_hist(
        axes[0],
        log_r,
        bins=85,
        color=color,
        label=label,
        alpha=0.16,
        linewidth=2.15,
    )

    plot_ecdf(
        axes[1],
        log_r,
        color=color,
        label=label,
        linewidth=2.2,
    )

polish_axes(
    axes[0],
    r"Density of $\log_{10}(R)$",
    r"$\log_{10}(R)$",
    "Density",
    legend=True,
)

polish_axes(
    axes[1],
    "ECDF",
    r"$\log_{10}(R)$",
    "Share of observations",
    legend=True,
)

plt.tight_layout()
plt.show()
```

## Noise Application


```python
def add_gaussian_noise_to_targets(
    y: pd.DataFrame,
    noise_level: float,
    seed: int,
) -> pd.DataFrame:
    if noise_level <= 0:
        return y.copy()

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()

    scale = y.std(axis=0, ddof=0).replace(0.0, 1.0)
    noise = rng.normal(loc=0.0, scale=1.0, size=y_noisy.shape)

    y_noisy.loc[:, :] = y_noisy.to_numpy() + noise_level * scale.to_numpy() * noise
    return y_noisy


if cfg.add_noise:
    y_train_raw = add_gaussian_noise_to_targets(
        y=y_train_raw,
        noise_level=cfg.noise_level,
        seed=cfg.noise_seed,
    )
```

## Normalization


```python
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train_raw)
X_valid = x_scaler.transform(X_valid_raw)
X_test = x_scaler.transform(X_test_raw)

y_train = y_scaler.fit_transform(y_train_raw)
y_valid = y_scaler.transform(y_valid_raw)
y_test = y_scaler.transform(y_test_raw)
```

## Data Loader


```python
def to_float_tensor(array_like) -> torch.Tensor:
    return torch.tensor(np.asarray(array_like), dtype=torch.float32)


X_train_t = to_float_tensor(X_train)
X_valid_t = to_float_tensor(X_valid)
X_test_t = to_float_tensor(X_test)

y_train_t = to_float_tensor(y_train)
y_valid_t = to_float_tensor(y_valid)
phys_train_t = to_float_tensor(phys_train_raw.values)
phys_valid_t = to_float_tensor(phys_valid_raw.values)
phys_test_t = to_float_tensor(phys_test_raw.values)

if cfg.use_weighted_sampler:
    train_R = phys_train_raw["R"].to_numpy()
    train_log_R = np.log1p(train_R)

    train_bins = pd.qcut(
        train_log_R,
        q=cfg.n_strat_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)

    bin_counts = np.bincount(train_bins)
    inverse_bin_weights = 1.0 / np.sqrt(bin_counts[train_bins])

    high_R_boost = 1.0 + cfg.sampler_high_R_scale * np.sqrt(np.clip(train_R, 0.0, 1.0))
    sample_weights = inverse_bin_weights * high_R_boost
    sample_weights = sample_weights / np.mean(sample_weights)

    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
else:
    train_sampler = None

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t, phys_train_t),
    batch_size=cfg.batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    drop_last=False,
)

valid_loader = DataLoader(
    TensorDataset(X_valid_t, y_valid_t, phys_valid_t),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)
```

## PINN Architecture


```python
class PhysicsInformedNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```


```python
model = PhysicsInformedNN(
    input_dim=len(feature_cols),
    output_dim=len(target_cols),
    hidden_dim=cfg.hidden_dim,
    num_hidden_layers=cfg.num_hidden_layers,
    dropout=cfg.dropout,
).to(device)

print(model)
print(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Physics


```python
PHYS = {name: i for i, name in enumerate(phys_cols)}

y_mean_t = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
y_scale_t = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)


def inverse_transform_y(y_scaled: torch.Tensor) -> torch.Tensor:
    return y_scaled * y_scale_t + y_mean_t


def as_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.complex(real, imag)


def unpack_zeta(y_physical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zeta1 = as_complex(y_physical[:, 0], y_physical[:, 1])
    zeta2 = as_complex(y_physical[:, 2], y_physical[:, 3])
    return zeta1, zeta2


def unpack_physics(phys_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    phys_batch = phys_batch.to(device)

    return {
        "psi": phys_batch[:, PHYS["psi"]],
        "delta_y": phys_batch[:, PHYS["delta_y"]],
        "kappa": phys_batch[:, PHYS["kappa"]],
        "mu": as_complex(phys_batch[:, PHYS["Re_mu"]], phys_batch[:, PHYS["Im_mu"]]),
        "S": as_complex(phys_batch[:, PHYS["Re_S"]], phys_batch[:, PHYS["Im_S"]]),
        "S1": as_complex(phys_batch[:, PHYS["Re_S1"]], phys_batch[:, PHYS["Im_S1"]]),
        "S2": as_complex(phys_batch[:, PHYS["Re_S2"]], phys_batch[:, PHYS["Im_S2"]]),
        "R_true": phys_batch[:, PHYS["R"]],
    }


def project_R(R: torch.Tensor, cfg: Config) -> torch.Tensor:
    return torch.clamp(R, min=cfg.min_R, max=cfg.max_R)


def reflection_coefficient(y_physical: torch.Tensor, phys_batch: torch.Tensor, cfg: Config) -> torch.Tensor:
    zeta1, zeta2 = unpack_zeta(y_physical)
    phys = unpack_physics(phys_batch)

    ky = phys["kappa"] * torch.sin(phys["psi"])
    phase = torch.exp(1j * ky * phys["delta_y"])

    amplitude = zeta1 + zeta2 * phase
    denominator = 4.0 * cfg.period_a * phys["kappa"] ** 3 * torch.sin(phys["psi"])
    denominator = torch.clamp(denominator, min=cfg.numerical_eps)

    R0 = 1j * phys["mu"] * amplitude / denominator.to(torch.complex64)
    R = torch.abs(R0) ** 2

    return project_R(R.float(), cfg)


def physics_loss(y_physical: torch.Tensor, phys_batch: torch.Tensor) -> torch.Tensor:
    zeta1, zeta2 = unpack_zeta(y_physical)
    phys = unpack_physics(phys_batch)

    ky = phys["kappa"] * torch.sin(phys["psi"])
    exp1 = torch.ones_like(zeta1)
    exp2 = torch.exp(1j * ky * phys["delta_y"])

    term11 = phys["mu"] * phys["S"] * zeta1
    term12 = phys["mu"] * phys["S1"] * zeta2
    term21 = phys["mu"] * phys["S2"] * zeta1
    term22 = phys["mu"] * phys["S"] * zeta2

    res1 = zeta1 - exp1 - term11 - term12
    res2 = zeta2 - exp2 - term21 - term22

    scale1 = 1.0 + torch.abs(exp1) + torch.abs(term11) + torch.abs(term12)
    scale2 = 1.0 + torch.abs(exp2) + torch.abs(term21) + torch.abs(term22)

    res1_norm = res1 / scale1
    res2_norm = res2 / scale2

    return torch.mean(torch.abs(res1_norm) ** 2 + torch.abs(res2_norm) ** 2)


def coefficient_loss(y_physical: torch.Tensor, phys_batch: torch.Tensor, cfg: Config) -> torch.Tensor:
    phys = unpack_physics(phys_batch)

    R_pred = reflection_coefficient(y_physical, phys_batch, cfg)
    R_true = project_R(phys["R_true"], cfg)

    eps = cfg.R_loss_eps
    loss_log = torch.mean((torch.log(R_pred + eps) - torch.log(R_true + eps)) ** 2)
    loss_sqrt = torch.mean((torch.sqrt(R_pred + eps) - torch.sqrt(R_true + eps)) ** 2)

    return cfg.R_loss_log_weight * loss_log + cfg.R_loss_sqrt_weight * loss_sqrt


data_loss_fn = nn.SmoothL1Loss(beta=0.5)
```

## Training & Loss Functions


```python
def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def total_loss_from_batch(
    model: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    bphys: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_scaled = model(bx)
    pred_physical = inverse_transform_y(pred_scaled)

    loss_data = data_loss_fn(pred_scaled, by)
    loss_phys = physics_loss(pred_physical, bphys)
    loss_R = coefficient_loss(pred_physical, bphys, cfg)

    loss = loss_data + cfg.lambda_phys * loss_phys + cfg.lambda_R * loss_R

    metrics = {
        "loss": float(loss.detach().cpu()),
        "data": float(loss_data.detach().cpu()),
        "physics": float(loss_phys.detach().cpu()),
        "R": float(loss_R.detach().cpu()),
    }
    return loss, metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)

    totals = {"loss": 0.0, "data": 0.0, "physics": 0.0, "R": 0.0}
    n_obs = 0

    for bx, by, bphys in loader:
        bx = bx.to(device)
        by = by.to(device)
        bphys = bphys.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        loss, metrics = total_loss_from_batch(model, bx, by, bphys)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

        batch_size = bx.shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        n_obs += batch_size

    return {key: value / n_obs for key, value in totals.items()}


def evaluate_loader(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    with torch.no_grad():
        return run_epoch(model, loader, optimizer=None)
```

## Training AdamW


```python
history = []
lbfgs_trace = []
checkpoint = None
model_was_loaded = False

if cfg.load_existing_model:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}\n")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_state_dict = clone_state_dict_to_cpu(model)
    best_valid = float(checkpoint.get("metrics", {}).get("selected_valid_loss", np.nan))
    best_epoch = -1
    best_stage = checkpoint.get("metrics", {}).get("selected_stage", "loaded")
    model_was_loaded = True

    print(f"Loaded model: {checkpoint_path}")
    print(f"Selected stage: {best_stage}")
    if np.isfinite(best_valid):
        print(f"Saved validation loss: {best_valid:.6e}")
    print("Training skipped")
else:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    best_valid = float("inf")
    best_epoch = -1
    best_stage = "init"
    best_state_dict = clone_state_dict_to_cpu(model)
    bad_epochs = 0

    adam_bar = tqdm(
        range(1, cfg.max_epochs + 1),
        desc="AdamW",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in adam_bar:
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer)
        valid_metrics = evaluate_loader(model, valid_loader)

        scheduler.step(valid_metrics["loss"])

        row = {
            "stage": "adam",
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if valid_metrics["loss"] < best_valid - cfg.min_delta:
            best_valid = valid_metrics["loss"]
            best_epoch = epoch
            best_stage = "adam"
            best_state_dict = clone_state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        adam_bar.set_postfix(
            train=f"{train_metrics['loss']:.2e}",
            valid=f"{valid_metrics['loss']:.2e}",
            phys=f"{valid_metrics['physics']:.2e}",
            logR=f"{valid_metrics['R']:.2e}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

        if bad_epochs >= cfg.patience:
            tqdm.write(f"Stopping at Epoch {epoch}. Best Epoch: {best_epoch}")
            break

    model.load_state_dict(best_state_dict)
    print(f"Best Adam Validation Loss: {best_valid:.6e} at Epoch {best_epoch}")

history_df = pd.DataFrame(history)
```

## Plot AdamW


```python
adam_history = history_df[history_df["stage"] == "adam"].copy() if "stage" in history_df.columns else pd.DataFrame()

if len(adam_history) == 0:
    print("AdamW plot skipped")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), dpi=150)
    fig.suptitle("Training AdamW", fontsize=15.5, fontweight="semibold", y=1.04)

    epoch = adam_history["epoch"].to_numpy()
    train_loss = adam_history["train_loss"].to_numpy()
    valid_loss = adam_history["valid_loss"].to_numpy()

    axes[0].plot(epoch, train_loss, color=COLORS["train"], alpha=0.22, linewidth=1.0)
    axes[0].plot(epoch, valid_loss, color=COLORS["valid"], alpha=0.22, linewidth=1.0)
    axes[0].plot(epoch, smooth_series(train_loss, 35), color=COLORS["train"], linewidth=2.4, label="Train loss")
    axes[0].plot(epoch, smooth_series(valid_loss, 35), color=COLORS["valid"], linewidth=2.4, label="Validation loss")
    axes[0].axvline(
        adam_history.loc[adam_history["valid_loss"].idxmin(), "epoch"],
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Best validation",
    )
    axes[0].set_yscale("log")
    polish_axes(axes[0], "Total loss", "Epoch", "Loss", legend=True)

    component_specs = [
        ("valid_data", "Data", COLORS["data"]),
        ("valid_physics", "Physics", COLORS["physics"]),
        ("valid_R", "Reflection", COLORS["reflection"]),
    ]

    for col, label, color in component_specs:
        values = adam_history[col].to_numpy()
        axes[1].plot(epoch, values, color=color, alpha=0.20, linewidth=1.0)
        axes[1].plot(epoch, smooth_series(values, 35), color=color, linewidth=2.2, label=label)

    axes[1].set_yscale("log")
    polish_axes(axes[1], "Validation loss", "Epoch", "Loss", legend=True)

    plt.tight_layout()
    plt.show()

    display(adam_history.tail())
```

## Training L-BFGS


```python
if cfg.use_lbfgs and not cfg.load_existing_model:
    model.load_state_dict(best_state_dict)
    model.train()

    X_train_full = X_train_t.to(device)
    y_train_full = y_train_t.to(device)
    phys_train_full = phys_train_t.to(device)

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.lbfgs_lr,
        max_iter=cfg.lbfgs_max_iter,
        max_eval=cfg.lbfgs_max_eval,
        history_size=cfg.lbfgs_history_size,
        tolerance_grad=cfg.lbfgs_tolerance_grad,
        tolerance_change=cfg.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )

    closure_calls = [0]

    lbfgs_best_valid = [best_valid]
    lbfgs_best_eval = [None]
    lbfgs_best_state_dict = [None]

    lbfgs_bar = tqdm(
        total=cfg.lbfgs_max_eval,
        desc="L-BFGS",
        unit="eval",
        dynamic_ncols=True,
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)

        loss, metrics = total_loss_from_batch(
            model,
            X_train_full,
            y_train_full,
            phys_train_full,
        )
        loss.backward()

        closure_calls[0] += 1

        should_log_train = closure_calls[0] == 1 or closure_calls[0] % cfg.lbfgs_log_every == 0
        should_log_valid = closure_calls[0] == 1 or closure_calls[0] % cfg.lbfgs_valid_log_every == 0

        valid_snapshot = {
            "loss": np.nan,
            "data": np.nan,
            "physics": np.nan,
            "R": np.nan,
        }

        if should_log_valid:
            model.eval()
            valid_snapshot = evaluate_loader(model, valid_loader)
            model.train()

            if valid_snapshot["loss"] < lbfgs_best_valid[0] - cfg.min_delta:
                lbfgs_best_valid[0] = valid_snapshot["loss"]
                lbfgs_best_eval[0] = closure_calls[0]
                lbfgs_best_state_dict[0] = clone_state_dict_to_cpu(model)

        if should_log_train or should_log_valid:
            lbfgs_trace.append({
                "eval": closure_calls[0],
                "train_loss": metrics["loss"],
                "train_data": metrics["data"],
                "train_physics": metrics["physics"],
                "train_R": metrics["R"],
                "valid_loss": valid_snapshot["loss"],
                "valid_data": valid_snapshot["data"],
                "valid_physics": valid_snapshot["physics"],
                "valid_R": valid_snapshot["R"],
            })

        lbfgs_bar.update(1)

        if closure_calls[0] == 1 or closure_calls[0] % 10 == 0:
            postfix = {
                "train": f"{metrics['loss']:.2e}",
                "phys": f"{metrics['physics']:.2e}",
                "logR": f"{metrics['R']:.2e}",
            }
            if should_log_valid:
                postfix["valid"] = f"{valid_snapshot['loss']:.2e}"
            lbfgs_bar.set_postfix(**postfix)

        return loss

    lbfgs.step(closure)
    lbfgs_bar.close()

    train_metrics = evaluate_loader(model, train_loader)
    valid_metrics = evaluate_loader(model, valid_loader)

    row = {
        "stage": "lbfgs",
        "epoch": closure_calls[0],
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"valid_{k}": v for k, v in valid_metrics.items()},
        "lr": cfg.lbfgs_lr,
    }
    history.append(row)

    print(f"L-BFGS Final Train Loss: {train_metrics['loss']:.6e}")
    print(f"L-BFGS Final Valid Loss: {valid_metrics['loss']:.6e}")

    if valid_metrics["loss"] < lbfgs_best_valid[0] - cfg.min_delta:
        lbfgs_best_valid[0] = valid_metrics["loss"]
        lbfgs_best_eval[0] = closure_calls[0]
        lbfgs_best_state_dict[0] = clone_state_dict_to_cpu(model)

    if lbfgs_best_state_dict[0] is not None:
        best_valid = lbfgs_best_valid[0]
        best_epoch = lbfgs_best_eval[0]
        best_stage = "lbfgs"
        best_state_dict = lbfgs_best_state_dict[0]
    else:
        print("L-BFGS did not improve")

elif cfg.load_existing_model:
    print("L-BFGS skipped")

elif not cfg.use_lbfgs:
    print("L-BFGS skipped")

model.load_state_dict(best_state_dict)
history_df = pd.DataFrame(history)

print(f"\nSelected Stage: {best_stage}")
print(f"Best Validation Loss After All Stages: {best_valid:.6e}")
```

## Plot L-BFGS


```python
lbfgs_trace_df = pd.DataFrame(lbfgs_trace)

if len(lbfgs_trace_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), dpi=150)
    fig.suptitle("Training L-BFGS", fontsize=15.5, fontweight="semibold", y=1.04)

    valid_trace = lbfgs_trace_df.dropna(subset=["valid_loss"])

    axes[0].plot(
        lbfgs_trace_df["eval"],
        lbfgs_trace_df["train_loss"],
        label="Train loss",
        color=COLORS["lbfgs"],
        linewidth=2.2,
    )

    if len(valid_trace) > 0:
        axes[0].plot(
            valid_trace["eval"],
            valid_trace["valid_loss"],
            label="Validation loss",
            color=COLORS["valid"],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )

        best_lbfgs_row = valid_trace.loc[valid_trace["valid_loss"].idxmin()]
        axes[0].axvline(
            best_lbfgs_row["eval"],
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            label="Best L-BFGS validation",
        )

    axes[0].set_yscale("log")
    polish_axes(axes[0], "Total loss", "Evaluation", "Loss", legend=True)

    if len(valid_trace) > 0:
        valid_components = [
            ("valid_data", "Data", COLORS["data"]),
            ("valid_physics", "Physics", COLORS["physics"]),
            ("valid_R", "Reflection", COLORS["reflection"]),
        ]

        for col, label, color in valid_components:
            axes[1].plot(
                valid_trace["eval"],
                valid_trace[col],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=2.8,
                label=label,
            )

    axes[1].set_yscale("log")
    polish_axes(axes[1], "Loss components", "Evaluation", "Loss", legend=True)

    plt.tight_layout()
    plt.show()

else:
    print("L-BFGS plot skipped")

display(history_df.tail())
```

## Test Prediction


```python
@torch.no_grad()
def predict_dataset(model, X_t: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = []

    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    for (bx,) in loader:
        bx = bx.to(device)
        pred_scaled = model(bx)
        pred_physical = inverse_transform_y(pred_scaled)
        preds.append(pred_physical.cpu().numpy())

    return np.vstack(preds)


@torch.no_grad()
def predict_R_dataset(
    model,
    X_t: torch.Tensor,
    phys_t: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    preds = []

    loader = DataLoader(TensorDataset(X_t, phys_t), batch_size=batch_size, shuffle=False)
    for bx, bphys in loader:
        bx = bx.to(device)
        bphys = bphys.to(device)

        pred_scaled = model(bx)
        pred_physical = inverse_transform_y(pred_scaled)
        R_pred = reflection_coefficient(pred_physical, bphys, cfg)

        preds.append(R_pred.cpu().numpy())

    return np.concatenate(preds)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    denom_mae = max(float(np.mean(np.abs(y_true))), eps)
    denom_rmse = max(float(np.sqrt(np.mean(np.asarray(y_true) ** 2))), eps)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "rMAE": mae / denom_mae,
        "rRMSE": rmse / denom_rmse,
        "R2": r2_score(y_true, y_pred),
    }


def regression_metrics_masked(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "rMAE": np.nan,
            "rRMSE": np.nan,
            "R2": np.nan,
            "count": 0,
            "share": 0.0,
        }

    metrics = regression_metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask], eps=eps)
    metrics["count"] = int(mask.sum())
    metrics["share"] = float(mask.mean())
    return metrics


y_pred_test = predict_dataset(model, X_test_t)
R_pred_test = predict_R_dataset(model, X_test_t, phys_test_t)

y_true_test = y_test_raw.values
R_true_test = phys_test_raw["R"].values
```

## Metrics


```python
rows = []

for i, col in enumerate(target_cols):
    metrics = regression_metrics(y_true_test[:, i], y_pred_test[:, i])
    rows.append({
        "Target": col,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "rMAE, %": 100.0 * metrics["rMAE"],
        "rRMSE, %": 100.0 * metrics["rRMSE"],
        "R2": metrics["R2"],
    })

metrics_R = regression_metrics(R_true_test, R_pred_test)
metrics_logR = regression_metrics(np.log1p(R_true_test), np.log1p(R_pred_test))

rows.append({
    "Target": "R (all)",
    "MAE": metrics_R["MAE"],
    "RMSE": metrics_R["RMSE"],
    "rMAE, %": 100.0 * metrics_R["rMAE"],
    "rRMSE, %": 100.0 * metrics_R["rRMSE"],
    "R2": metrics_R["R2"],
})

for min_R_value in [cfg.relative_error_min_R, 1.0e-3, 1.0e-2]:
    mask = R_true_test >= min_R_value
    metrics_masked = regression_metrics_masked(R_true_test, R_pred_test, mask)
    rows.append({
        "Target": rf"R ($R \geq {min_R_value:g}$)",
        "MAE": metrics_masked["MAE"],
        "RMSE": metrics_masked["RMSE"],
        "rMAE, %": 100.0 * metrics_masked["rMAE"],
        "rRMSE, %": 100.0 * metrics_masked["rRMSE"],
        "R2": metrics_masked["R2"],
    })

rows.append({
    "Target": "log1p(R) (all)",
    "MAE": metrics_logR["MAE"],
    "RMSE": metrics_logR["RMSE"],
    "rMAE, %": 100.0 * metrics_logR["rMAE"],
    "rRMSE, %": 100.0 * metrics_logR["rRMSE"],
    "R2": metrics_logR["R2"],
})

metrics_df = pd.DataFrame(rows)
display(metrics_df)


metrics_df.to_csv(visualization_run_dir / "metrics.csv", index=False)
save_dataframe_text(metrics_df, visualization_run_dir / "metrics.txt")
save_dataframe_as_png(metrics_df, visualization_run_dir / "metrics.png", title="Test Metrics")
```

## Prediction Visualization


```python
rng = np.random.default_rng(SEED)
plot_size = min(cfg.plot_sample_size, len(R_true_test))
plot_idx = rng.choice(len(R_true_test), size=plot_size, replace=False)

error_R = R_pred_test - R_true_test
abs_error_R = np.abs(error_R)

mask_rel = R_true_test >= cfg.relative_error_min_R
rel_error_R_pct_all = np.full_like(R_true_test, np.nan, dtype=float)
rel_error_R_pct_all[mask_rel] = 100.0 * abs_error_R[mask_rel] / R_true_test[mask_rel]

zeta1_true = y_true_test[:, 0] + 1j * y_true_test[:, 1]
zeta1_pred = y_pred_test[:, 0] + 1j * y_pred_test[:, 1]
zeta2_true = y_true_test[:, 2] + 1j * y_true_test[:, 3]
zeta2_pred = y_pred_test[:, 2] + 1j * y_pred_test[:, 3]

zeta_eps = cfg.zeta_error_eps
zeta1_error_pct = 100.0 * np.abs(zeta1_pred - zeta1_true) / np.maximum(np.abs(zeta1_true), zeta_eps)
zeta2_error_pct = 100.0 * np.abs(zeta2_pred - zeta2_true) / np.maximum(np.abs(zeta2_true), zeta_eps)

fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), dpi=150)
fig.suptitle("Prediction Quality", fontsize=16, fontweight="semibold", y=1.02)

hb = axes[0, 0].hexbin(
    R_true_test[plot_idx],
    R_pred_test[plot_idx],
    gridsize=58,
    bins="log",
    mincnt=1,
    cmap="viridis",
)
axes[0, 0].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.6, color=COLORS["ideal"], label="Perfect prediction")
axes[0, 0].set_xlim(0.0, 1.0)
axes[0, 0].set_ylim(0.0, 1.0)
polish_axes(
    axes[0, 0],
    "Predicted vs True $R$",
    r"$R_{true}$",
    r"$R_{pred}$",
    legend=True,
)
cb = fig.colorbar(hb, ax=axes[0, 0], fraction=0.046, pad=0.04)
cb.set_label("log(count)", labelpad=8)

rel_errors = rel_error_R_pct_all[np.isfinite(rel_error_R_pct_all)]
rel_sorted = np.sort(rel_errors)
rel_ecdf = np.linspace(0.0, 100.0, len(rel_sorted), endpoint=True)

axes[0, 1].plot(
    rel_sorted,
    rel_ecdf,
    color=COLORS["lbfgs"],
    linewidth=2.4,
    label="Relative Error ECDF",
)
for threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
    if threshold <= np.nanmax(rel_sorted):
        axes[0, 1].axvline(threshold, linestyle="--", linewidth=1.0, alpha=0.6, color=COLORS["gray"])
axes[0, 1].set_xscale("log")
axes[0, 1].set_ylim(0, 101)
polish_axes(
    axes[0, 1],
    "ECDF of $R$ relative error",
    "Relative error $R$, %",
    "Share of samples, %",
    legend=True,
)

plot_mask = np.isfinite(rel_error_R_pct_all[plot_idx])
axes[1, 0].scatter(
    R_true_test[plot_idx][plot_mask],
    rel_error_R_pct_all[plot_idx][plot_mask],
    s=10,
    alpha=0.22,
    color=COLORS["train"],
    edgecolors="none",
    label="Test points",
)

bins_for_trend = np.geomspace(max(cfg.relative_error_min_R, 1e-8), 1.0, 18)
bin_ids = np.digitize(R_true_test[mask_rel], bins_for_trend)
trend_x, trend_y = [], []
for b in range(1, len(bins_for_trend)):
    m = bin_ids == b
    if np.sum(m) >= 20:
        trend_x.append(np.sqrt(bins_for_trend[b - 1] * bins_for_trend[b]))
        trend_y.append(np.nanmedian(rel_error_R_pct_all[mask_rel][m]))

if len(trend_x) > 0:
    axes[1, 0].plot(
        trend_x,
        trend_y,
        color=COLORS["ideal"],
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="Median by $R$ bin",
    )

axes[1, 0].set_xscale("log")
axes[1, 0].set_yscale("log")
polish_axes(
    axes[1, 0],
    "$R$ error by true value",
    r"$R_{true}$",
    "Relative error, %",
    legend=True,
)

quantities = ["R", r"|ζ1|", r"|ζ2|"]
mean_errors = np.array([
    np.nanmean(rel_error_R_pct_all),
    np.mean(zeta1_error_pct),
    np.mean(zeta2_error_pct),
])

bar_colors = [COLORS["reflection"], COLORS["data"], COLORS["physics"]]
axes[1, 1].barh(
    quantities[::-1],
    mean_errors[::-1],
    color=bar_colors[::-1],
    alpha=0.88,
    edgecolor="white",
    linewidth=0.7,
)
axes[1, 1].set_xscale("log")

for i, value in enumerate(mean_errors[::-1]):
    axes[1, 1].text(
        value * 1.08,
        i,
        f"{value:.3g}%",
        va="center",
        ha="left",
        fontsize=10,
        color="#111827",
    )

polish_axes(
    axes[1, 1],
    "Mean percentage error by quantity",
    "Mean error, %",
    "Quantity",
)

plt.tight_layout()
save_current_figure(visualization_run_dir / "prediction_quality.png")
plt.show()
```

## Relative Error


```python
global_relative_metrics_df = pd.DataFrame({
    "Target": ["R"],
    "rMAE, %": [100.0 * metrics_R["rMAE"]],
    "rRMSE, %": [100.0 * metrics_R["rRMSE"]],
})

rel_R = rel_error_R_pct_all[np.isfinite(rel_error_R_pct_all)]

thresholds = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
coverage = np.array([
    100.0 * np.mean(rel_R <= threshold)
    for threshold in thresholds
])

coverage_df = pd.DataFrame({
    "threshold_percent": thresholds,
    "share_of_test_points_percent": coverage,
})

error_by_R = pd.DataFrame({
    "R_true": R_true_test[mask_rel],
    "relative_error_percent": rel_error_R_pct_all[mask_rel],
})

R_bins = [cfg.relative_error_min_R, 1e-3, 1e-2, 1e-1, np.inf]
R_bin_labels = [
    r"$[10^{-4}, 10^{-3}]$",
    r"$(10^{-3}, 10^{-2}]$",
    r"$(10^{-2}, 10^{-1}]$",
    r"$(10^{-1}, 1]$",
]

error_by_R["R_range"] = pd.cut(
    error_by_R["R_true"],
    bins=R_bins,
    labels=R_bin_labels,
    include_lowest=True,
)

range_summary = (
    error_by_R
    .groupby("R_range", observed=True)
    .agg(
        count=("relative_error_percent", "size"),
        mean_error_percent=("relative_error_percent", "mean"),
        share_error_below_1pct=("relative_error_percent", lambda x: 100.0 * np.mean(x <= 1.0)),
    )
    .reset_index()
)

display(range_summary)

range_summary.to_csv(visualization_run_dir / "relative_error_range_summary.csv", index=False)
save_dataframe_text(range_summary, visualization_run_dir / "relative_error_range_summary.txt")

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.1), dpi=150)
fig.suptitle(r"Estimate of $R$ Relative Error", fontsize=15.5, fontweight="semibold", y=1.04)

bars = axes[0].bar(
    [f"≤ {x:g}%" for x in thresholds],
    coverage,
    color=[COLORS["train"], COLORS["valid"], COLORS["test"], COLORS["lbfgs"], COLORS["amber"]],
    alpha=0.88,
    edgecolor="white",
    linewidth=0.8,
)
axes[0].set_ylim(0, 105)

for bar, value in zip(bars, coverage):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        min(value + 2.0, 103),
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#111827",
    )

polish_axes(
    axes[0],
    "Share of test points below error threshold",
    "Relative error threshold",
    "Share of samples, %",
)

x = np.arange(len(range_summary))
width = 0.36

axes[1].bar(
    x - width / 2,
    range_summary["mean_error_percent"],
    width=width,
    color=COLORS["reflection"],
    alpha=0.88,
    edgecolor="white",
    linewidth=0.8,
    label=r"Mean $R$ error, %",
)

axes[1].bar(
    x + width / 2,
    range_summary["share_error_below_1pct"],
    width=width,
    color=COLORS["purple"],
    alpha=0.78,
    edgecolor="white",
    linewidth=0.8,
    label="Error ≤ 1%, share of samples",
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(range_summary["R_range"].astype(str), rotation=18, ha="right")
axes[1].set_yscale("log")

polish_axes(
    axes[1],
    r"Error by true $R$ range",
    r"$R$ range",
    "Percent, %",
    legend=True,
)

plt.tight_layout()
save_current_figure(visualization_run_dir / "relative_error.png")
plt.show()
```

## Reflection Spectrum


```python
@dataclass(frozen=True)
class SpectrumPlateParams:
    rho: float
    h: float
    E: float
    nu: float

    @property
    def D(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))


@dataclass(frozen=True)
class SpectrumResonatorParams:
    omega0: float
    damping_ratio: float
    mu_scale: float

    @property
    def gamma(self) -> float:
        return self.damping_ratio * self.omega0


def omega_from_kappa(kappa_value: np.ndarray, plate: SpectrumPlateParams) -> np.ndarray:
    return np.sqrt((np.asarray(kappa_value) ** 4) * plate.D / (plate.rho * plate.h))


def mu_resonator_np(omega: np.ndarray, resonator: SpectrumResonatorParams) -> np.ndarray:
    denominator = resonator.omega0**2 - omega**2 - 1j * resonator.gamma * omega
    return -resonator.mu_scale * omega**2 / denominator


def lambda_gamma_np(kxn: np.ndarray, kappa_value: float) -> tuple[np.ndarray, np.ndarray]:
    ratio = kxn / kappa_value
    ratio2 = ratio * ratio

    lam = np.empty_like(ratio, dtype=np.complex128)
    propagating = np.abs(ratio) < 1.0

    lam[propagating] = -1j * np.sqrt(1.0 - ratio2[propagating])
    lam[~propagating] = np.sqrt(ratio2[~propagating] - 1.0) + 0.0j

    gam = np.sqrt(ratio2 + 1.0).astype(np.complex128)
    return lam, gam


def lattice_sums_np(
    psi: float,
    delta_x: float,
    delta_y: float,
    kappa_value: float,
    period_a: float,
    n_terms: int = 80,
    eps: float = 1.0e-12,
) -> tuple[complex, complex, complex]:
    n_values = np.arange(-n_terms, n_terms + 1, dtype=np.float64)
    kx = kappa_value * np.cos(psi)
    kxn = kx + 2.0 * np.pi * n_values / period_a

    lam, gam = lambda_gamma_np(kxn, kappa_value)
    valid = (np.abs(lam) >= eps) & (np.abs(gam) >= eps)

    inv_term = np.zeros_like(lam, dtype=np.complex128)
    inv_term[valid] = 1.0 / lam[valid] - 1.0 / gam[valid]

    norm = 4.0 * period_a * kappa_value**3
    S = np.sum(inv_term) / norm

    dy = abs(delta_y)
    cross_term = np.zeros_like(lam, dtype=np.complex128)
    cross_term[valid] = (
        np.exp(-kappa_value * lam[valid] * dy) / lam[valid]
        - np.exp(-kappa_value * gam[valid] * dy) / gam[valid]
    )

    phase = np.exp(1j * kx * delta_x)
    S1 = np.conj(phase) * np.sum(cross_term) / norm
    S2 = phase * np.sum(cross_term) / norm

    return complex(S), complex(S1), complex(S2)


def solve_zeta_np(
    psi: float,
    delta_y: float,
    mu: complex,
    S: complex,
    S1: complex,
    S2: complex,
    kappa_value: float,
) -> tuple[complex, complex]:
    ky = kappa_value * np.sin(psi)

    exp1 = 1.0 + 0.0j
    exp2 = np.exp(1j * ky * delta_y)

    denominator = (1.0 - mu * S) ** 2 - mu**2 * S1 * S2
    zeta1 = ((1.0 - mu * S) * exp1 + mu * S1 * exp2) / denominator
    zeta2 = ((1.0 - mu * S) * exp2 + mu * S2 * exp1) / denominator

    return zeta1, zeta2


def reflection_np(
    psi: float,
    delta_y: float,
    mu: complex,
    zeta1: complex,
    zeta2: complex,
    kappa_value: float,
    period_a: float,
) -> float:
    ky = kappa_value * np.sin(psi)
    amplitude = zeta1 + zeta2 * np.exp(1j * ky * delta_y)
    R0 = 1j * mu * amplitude / (4.0 * period_a * kappa_value**3 * np.sin(psi))
    return float(np.clip(abs(R0) ** 2, cfg.min_R, cfg.max_R))


def build_fixed_spectrum_df(
    psi: float,
    delta_x: float,
    delta_y: float,
    kappa_grid: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    plate = SpectrumPlateParams(
        rho=cfg.plate_rho,
        h=cfg.plate_h,
        E=cfg.plate_E,
        nu=cfg.plate_nu,
    )
    resonator = SpectrumResonatorParams(
        omega0=cfg.omega0,
        damping_ratio=cfg.resonator_damping_ratio,
        mu_scale=cfg.resonator_mu_scale,
    )

    omega_grid = omega_from_kappa(kappa_grid, plate)
    mu_grid = mu_resonator_np(omega_grid, resonator)

    rows = []
    for kap, omega, mu in zip(kappa_grid, omega_grid, mu_grid):
        S, S1, S2 = lattice_sums_np(
            psi=psi,
            delta_x=delta_x,
            delta_y=delta_y,
            kappa_value=float(kap),
            period_a=cfg.period_a,
            n_terms=cfg.spectrum_n_terms,
            eps=cfg.numerical_eps,
        )
        zeta1, zeta2 = solve_zeta_np(
            psi=psi,
            delta_y=delta_y,
            mu=mu,
            S=S,
            S1=S1,
            S2=S2,
            kappa_value=float(kap),
        )
        R = reflection_np(
            psi=psi,
            delta_y=delta_y,
            mu=mu,
            zeta1=zeta1,
            zeta2=zeta2,
            kappa_value=float(kap),
            period_a=cfg.period_a,
        )

        rows.append({
            "omega": float(omega),
            "psi": psi,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "kappa": float(kap),
            "Re_mu": mu.real,
            "Im_mu": mu.imag,
            "Re_S": S.real,
            "Im_S": S.imag,
            "Re_S1": S1.real,
            "Im_S1": S1.imag,
            "Re_S2": S2.real,
            "Im_S2": S2.imag,
            "R": R,
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_fixed_spectrum(spectrum_df: pd.DataFrame) -> np.ndarray:
    X_spectrum_raw = build_features(spectrum_df, cfg)[feature_cols]
    X_spectrum = torch.tensor(
        x_scaler.transform(X_spectrum_raw),
        dtype=torch.float32,
        device=device,
    )

    phys_spectrum = torch.tensor(
        spectrum_df[phys_cols].values,
        dtype=torch.float32,
        device=device,
    )

    return predict_R_dataset(model, X_spectrum, phys_spectrum)


kappa_grid = np.linspace(
    max(float(df["kappa"].min()), 1.0e-3),
    float(df["kappa"].max()),
    cfg.fixed_spectrum_points,
)

spectrum_cases = [
    {
        "title": r"Half-period array: $\delta_x=0.5a$, $\delta_y\approx0$, $\psi=\pi/6$",
        "psi": cfg.reference_psi,
        "delta_x": 0.5 * cfg.period_a,
        "delta_y": cfg.near_zero_delta_y,
    },
    {
        "title": r"Zigzag array: $\delta_x=0.5a$, $\delta_y=0.5a\tan(\pi/6)$, $\psi=\pi/6$",
        "psi": cfg.reference_psi,
        "delta_x": 0.5 * cfg.period_a,
        "delta_y": 0.5 * cfg.period_a * np.tan(cfg.reference_psi),
    },
]

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.0), dpi=150)
fig.suptitle(r"Fixed-Geometry Reflection Spectra", fontsize=15.5, fontweight="semibold", y=1.04)

for ax, case in zip(axes, spectrum_cases):
    spectrum_df = build_fixed_spectrum_df(
        psi=case["psi"],
        delta_x=case["delta_x"],
        delta_y=case["delta_y"],
        kappa_grid=kappa_grid,
        cfg=cfg,
    )
    R_spectrum_true = spectrum_df["R"].to_numpy()
    R_spectrum_pred = predict_fixed_spectrum(spectrum_df)

    ax.plot(
        spectrum_df["kappa"],
        R_spectrum_true,
        color=COLORS["ideal"],
        linewidth=2.3,
        label=r"$R_{true}$",
    )
    ax.plot(
        spectrum_df["kappa"],
        R_spectrum_pred,
        color=COLORS["reflection"],
        linewidth=2.0,
        linestyle="--",
        label=r"$R_{pred}$",
    )

    spectrum_metrics = regression_metrics(R_spectrum_true, R_spectrum_pred)
    ax.text(
        0.03,
        0.95,
        f"rMAE={100.0 * spectrum_metrics['rMAE']:.2f}%\n"
        f"rRMSE={100.0 * spectrum_metrics['rRMSE']:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": COLORS["light_gray"],
            "alpha": 0.92,
        },
    )

    polish_axes(
        ax,
        case["title"],
        r"$\kappa$",
        r"$|R_0|^2$",
        legend=True,
    )

plt.tight_layout()
save_current_figure(visualization_run_dir / "reflection_spectrum.png")
plt.show()
```

## Error Plot


```python
def build_geometry_heatmap_df(
    psi: float,
    kappa_value: float,
    delta_x_values: np.ndarray,
    delta_y_values: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    plate = SpectrumPlateParams(
        rho=cfg.plate_rho,
        h=cfg.plate_h,
        E=cfg.plate_E,
        nu=cfg.plate_nu,
    )
    resonator = SpectrumResonatorParams(
        omega0=cfg.omega0,
        damping_ratio=cfg.resonator_damping_ratio,
        mu_scale=cfg.resonator_mu_scale,
    )

    omega_value = float(omega_from_kappa(np.array([kappa_value]), plate)[0])
    mu = complex(mu_resonator_np(np.array([omega_value]), resonator)[0])

    rows = []
    for dy in delta_y_values:
        for dx in delta_x_values:
            S, S1, S2 = lattice_sums_np(
                psi=psi,
                delta_x=float(dx),
                delta_y=float(dy),
                kappa_value=float(kappa_value),
                period_a=cfg.period_a,
                n_terms=cfg.spectrum_n_terms,
                eps=cfg.numerical_eps,
            )
            zeta1, zeta2 = solve_zeta_np(
                psi=psi,
                delta_y=float(dy),
                mu=mu,
                S=S,
                S1=S1,
                S2=S2,
                kappa_value=float(kappa_value),
            )
            R = reflection_np(
                psi=psi,
                delta_y=float(dy),
                mu=mu,
                zeta1=zeta1,
                zeta2=zeta2,
                kappa_value=float(kappa_value),
                period_a=cfg.period_a,
            )

            rows.append({
                "omega": omega_value,
                "psi": psi,
                "delta_x": float(dx),
                "delta_y": float(dy),
                "kappa": float(kappa_value),
                "Re_mu": mu.real,
                "Im_mu": mu.imag,
                "Re_S": S.real,
                "Im_S": S.imag,
                "Re_S1": S1.real,
                "Im_S1": S1.imag,
                "Re_S2": S2.real,
                "Im_S2": S2.imag,
                "Re_zeta1": zeta1.real,
                "Im_zeta1": zeta1.imag,
                "Re_zeta2": zeta2.real,
                "Im_zeta2": zeta2.imag,
                "R": R,
            })

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_geometry_outputs(geometry_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_geometry_raw = build_features(geometry_df, cfg)[feature_cols]
    X_geometry = torch.tensor(
        x_scaler.transform(X_geometry_raw),
        dtype=torch.float32,
        device=device,
    )
    phys_geometry = torch.tensor(
        geometry_df[phys_cols].values,
        dtype=torch.float32,
        device=device,
    )

    y_pred = predict_dataset(model, X_geometry)
    R_pred = predict_R_dataset(model, X_geometry, phys_geometry)
    return y_pred, R_pred

def plot_exact_pred_abs_triplet(
    true_map: np.ndarray,
    pred_map: np.ndarray,
    abs_error_map: np.ndarray,
    title: str,
    true_label: str,
    pred_label: str,
    error_label: str,
    extent: list[float],
    cmap_main: str = "viridis",
    cmap_error: str = "magma",
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), dpi=150)
    fig.suptitle(title, fontsize=15.0, fontweight="semibold", y=1.04)

    finite_true_pred = np.concatenate([
        true_map[np.isfinite(true_map)].ravel(),
        pred_map[np.isfinite(pred_map)].ravel(),
    ])

    if finite_true_pred.size == 0:
        vmin_main, vmax_main = 0.0, 1.0
    else:
        vmin_main = float(np.nanpercentile(finite_true_pred, 1))
        vmax_main = float(np.nanpercentile(finite_true_pred, 99))
        if np.isclose(vmin_main, vmax_main):
            pad = max(abs(vmin_main), 1.0) * 1.0e-3
            vmin_main -= pad
            vmax_main += pad

    im0 = axes[0].imshow(
        true_map,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap_main,
        vmin=vmin_main,
        vmax=vmax_main,
    )
    polish_axes(axes[0], "Exact", r"$\delta_x/a$", r"$\delta_y/a$")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(true_label, labelpad=8)

    im1 = axes[1].imshow(
        pred_map,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap_main,
        vmin=vmin_main,
        vmax=vmax_main,
    )
    polish_axes(axes[1], "PINN prediction", r"$\delta_x/a$", r"$\delta_y/a$")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(pred_label, labelpad=8)

    err_vmax = float(np.nanquantile(abs_error_map[np.isfinite(abs_error_map)], 0.99)) if np.isfinite(abs_error_map).any() else None
    im2 = axes[2].imshow(
        abs_error_map,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap_error,
        vmin=0.0,
        vmax=err_vmax,
    )
    polish_axes(axes[2], "Absolute error", r"$\delta_x/a$", r"$\delta_y/a$")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label(error_label, labelpad=8)

    plt.tight_layout()
    if save_path is not None:
        save_current_figure(save_path)
    plt.show()
```


```python
q_low = df["R"].quantile(cfg.geometry_kappa_quantile_low)
q_high = df["R"].quantile(cfg.geometry_kappa_quantile_high)
moderate_high_R_part = df[(df["R"] >= q_low) & (df["R"] <= q_high)]
if len(moderate_high_R_part) > 0:
    heatmap_kappa = float(moderate_high_R_part["kappa"].median())
else:
    heatmap_kappa = float(df["kappa"].median())

heatmap_psi = cfg.reference_psi
heatmap_grid_size = cfg.geometry_grid_size

delta_x_values = np.linspace(0.0, cfg.period_a, heatmap_grid_size)
delta_y_values = np.linspace(float(df["delta_y"].min()), cfg.period_a, heatmap_grid_size)

geometry_df = build_geometry_heatmap_df(
    psi=heatmap_psi,
    kappa_value=heatmap_kappa,
    delta_x_values=delta_x_values,
    delta_y_values=delta_y_values,
    cfg=cfg,
)

y_true_geometry = geometry_df[target_cols].to_numpy()
y_pred_geometry, R_pred_geometry = predict_geometry_outputs(geometry_df)

extent = [
    delta_x_values.min() / cfg.period_a,
    delta_x_values.max() / cfg.period_a,
    delta_y_values.min() / cfg.period_a,
    delta_y_values.max() / cfg.period_a,
]

R_true_map = geometry_df["R"].to_numpy().reshape(heatmap_grid_size, heatmap_grid_size)
R_pred_map = R_pred_geometry.reshape(heatmap_grid_size, heatmap_grid_size)
R_abs_error_map = np.abs(R_pred_map - R_true_map)
zeta_true_maps = {
    col: y_true_geometry[:, i].reshape(heatmap_grid_size, heatmap_grid_size)
    for i, col in enumerate(target_cols)
}
zeta_pred_maps = {
    col: y_pred_geometry[:, i].reshape(heatmap_grid_size, heatmap_grid_size)
    for i, col in enumerate(target_cols)
}
zeta_abs_error_maps = {
    col: np.abs(zeta_pred_maps[col] - zeta_true_maps[col])
    for col in target_cols
}
heatmap_metrics = regression_metrics(R_true_map.ravel(), R_pred_map.ravel())

heatmap_metrics_df = pd.DataFrame([{
    "Target": "R heatmap",
    "MAE": heatmap_metrics["MAE"],
    "RMSE": heatmap_metrics["RMSE"],
    "rMAE, %": 100.0 * heatmap_metrics["rMAE"],
    "rRMSE, %": 100.0 * heatmap_metrics["rRMSE"],
    "R2": heatmap_metrics["R2"],
}])
heatmap_metrics_df.to_csv(visualization_run_dir / "heatmap_metrics.csv", index=False)
save_dataframe_text(heatmap_metrics_df, visualization_run_dir / "heatmap_metrics.txt")
```


```python
plot_exact_pred_abs_triplet(
    true_map=R_true_map,
    pred_map=R_pred_map,
    abs_error_map=R_abs_error_map,
    title=rf"Geometry map of $R(\delta_x,\delta_y)$ at $\psi=\pi/6$, $\kappa={heatmap_kappa:.3f}$",
    true_label=r"$R_{true}$",
    pred_label=r"$R_{pred}$",
    error_label=r"$|R_{pred}-R_{true}|$",
    extent=extent,
    save_path=visualization_run_dir / "heatmap_R.png",
)
```


```python
plot_exact_pred_abs_triplet(
    true_map=zeta_true_maps["Re_zeta1"],
    pred_map=zeta_pred_maps["Re_zeta1"],
    abs_error_map=zeta_abs_error_maps["Re_zeta1"],
    title=rf"Geometry map of $\mathrm{{Re}}\,\zeta_1$ at $\psi=\pi/6$, $\kappa={heatmap_kappa:.3f}$",
    true_label=r"$\mathrm{Re}\,\zeta_1$ exact",
    pred_label=r"$\mathrm{Re}\,\zeta_1$ predicted",
    error_label=r"$|\Delta \mathrm{Re}\,\zeta_1|$",
    extent=extent,
    save_path=visualization_run_dir / "heatmap_Re_zeta1.png",
)
```


```python
plot_exact_pred_abs_triplet(
    true_map=zeta_true_maps["Im_zeta1"],
    pred_map=zeta_pred_maps["Im_zeta1"],
    abs_error_map=zeta_abs_error_maps["Im_zeta1"],
    title=rf"Geometry map of $\mathrm{{Im}}\,\zeta_1$ at $\psi=\pi/6$, $\kappa={heatmap_kappa:.3f}$",
    true_label=r"$\mathrm{Im}\,\zeta_1$ exact",
    pred_label=r"$\mathrm{Im}\,\zeta_1$ predicted",
    error_label=r"$|\Delta \mathrm{Im}\,\zeta_1|$",
    extent=extent,
    save_path=visualization_run_dir / "heatmap_Im_zeta1.png",
)
```


```python
plot_exact_pred_abs_triplet(
    true_map=zeta_true_maps["Re_zeta2"],
    pred_map=zeta_pred_maps["Re_zeta2"],
    abs_error_map=zeta_abs_error_maps["Re_zeta2"],
    title=rf"Geometry map of $\mathrm{{Re}}\,\zeta_2$ at $\psi=\pi/6$, $\kappa={heatmap_kappa:.3f}$",
    true_label=r"$\mathrm{Re}\,\zeta_2$ exact",
    pred_label=r"$\mathrm{Re}\,\zeta_2$ predicted",
    error_label=r"$|\Delta \mathrm{Re}\,\zeta_2|$",
    extent=extent,
    save_path=visualization_run_dir / "heatmap_Re_zeta2.png",
)
```


```python
plot_exact_pred_abs_triplet(
    true_map=zeta_true_maps["Im_zeta2"],
    pred_map=zeta_pred_maps["Im_zeta2"],
    abs_error_map=zeta_abs_error_maps["Im_zeta2"],
    title=rf"Geometry map of $\mathrm{{Im}}\,\zeta_2$ at $\psi=\pi/6$, $\kappa={heatmap_kappa:.3f}$",
    true_label=r"$\mathrm{Im}\,\zeta_2$ exact",
    pred_label=r"$\mathrm{Im}\,\zeta_2$ predicted",
    error_label=r"$|\Delta \mathrm{Im}\,\zeta_2|$",
    extent=extent,
    save_path=visualization_run_dir / "heatmap_Im_zeta2.png",
)
```

## Physics Loss


```python
@torch.no_grad()
def physics_loss_dataset(model, X_t: torch.Tensor, phys_t: torch.Tensor, batch_size: int = 4096) -> float:
    model.eval()
    total = 0.0
    n_obs = 0

    loader = DataLoader(TensorDataset(X_t, phys_t), batch_size=batch_size, shuffle=False)
    for bx, bphys in loader:
        bx = bx.to(device)
        bphys = bphys.to(device)

        pred_scaled = model(bx)
        pred_physical = inverse_transform_y(pred_scaled)
        loss = physics_loss(pred_physical, bphys)

        total += loss.item() * bx.shape[0]
        n_obs += bx.shape[0]

    return total / n_obs


test_phys_loss = physics_loss_dataset(model, X_test_t, phys_test_t)
print(f"Test Physics Loss: {test_phys_loss:.6e}")
```

## Save Model


```python
def config_to_serializable_dict(cfg: Config) -> dict:
    result = {}
    for key, value in asdict(cfg).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


model.load_state_dict(best_state_dict)

checkpoint = {
    "model_state_dict": best_state_dict,
    "config": config_to_serializable_dict(cfg),
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "phys_cols": phys_cols,
    "architecture": "PhysicsInformedNN",
    "x_scaler_mean": x_scaler.mean_,
    "x_scaler_scale": x_scaler.scale_,
    "y_scaler_mean": y_scaler.mean_,
    "y_scaler_scale": y_scaler.scale_,
    "metrics": {
        "test_metrics": metrics_df.to_dict(orient="records"),
        "test_physics_loss": float(test_phys_loss),
        "selected_valid_loss": float(best_valid),
        "selected_stage": best_stage,
    },
    "noise": {
        "add_noise": cfg.add_noise,
        "noise_level": cfg.noise_level,
        "noise_percent": noise_percent,
        "noise_seed": cfg.noise_seed,
    },
    "n_objects": base_n_objects,
    "model_run_id": int(resolved_model_run_id),
    "visualization_run_dir": str(visualization_run_dir),
    "dataset_accounting": {
        "base_dataset_rows": int(base_n_objects),
        "base_train_rows": int(n_train_base),
        "spectrum_augmentation_rows": int(n_aug),
        "train_rows_used_by_optimizer": int(len(X_train_raw)),
        "validation_rows": int(len(idx_valid)),
        "test_rows": int(len(idx_test)),
        "augmentation_is_train_only": bool(n_aug > 0),
    },
    "seed": SEED,
}

if model_was_loaded:
    print(f"Loaded model without overwriting: {checkpoint_path}")
elif cfg.save_model_after_training:
    torch.save(checkpoint, model_path)
    print(f"Saved model: {model_path}")
else:
    print("Model saving skipped")
```
