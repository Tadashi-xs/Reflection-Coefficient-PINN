from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlateParams:
    rho: float = 7800.0
    h: float = 0.002
    E: float = 2.0e11
    nu: float = 0.3

    @property
    def D(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))


@dataclass(frozen=True)
class ResonatorParams:
    omega0: float = 2.0 * np.pi * 100.0
    damping_ratio: float = 0.01
    mu_scale: float = 1.0

    @property
    def gamma(self) -> float:
        return self.damping_ratio * self.omega0


@dataclass(frozen=True)
class LatticeParams:
    a: float = 0.05
    n_terms: int = 80


@dataclass(frozen=True)
class ModelParams:
    plate: PlateParams = PlateParams()
    resonator: ResonatorParams = ResonatorParams()
    lattice: LatticeParams = LatticeParams()


def omega_from_kappa(kappa_value: np.ndarray, plate: PlateParams) -> np.ndarray:
    return np.sqrt(plate.D * kappa_value**4 / (plate.rho * plate.h))


def mu_resonator(omega: np.ndarray, resonator: ResonatorParams) -> np.ndarray:
    denominator = resonator.omega0**2 - omega**2 - 1j * resonator.gamma * omega
    return -resonator.mu_scale * omega**2 / denominator


def make_logit_R(values: np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def resolve_geometry(
    geometry: str,
    period: float,
    psi: float,
    dx_fraction: float,
    dy_fraction: float,
) -> tuple[float, float]:
    if geometry == "half_period":
        return 0.5 * period, 0.0
    if geometry == "zigzag":
        return 0.5 * period, 0.5 * period * np.tan(psi)
    if geometry == "custom":
        return dx_fraction * period, dy_fraction * period
    raise ValueError(f"unknown geometry: {geometry}")


def compute_R_for_kappa_grid(
    kappa_values: np.ndarray,
    params: ModelParams,
    psi: float,
    delta_x: float,
    delta_y: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    kappa_values = np.asarray(kappa_values, dtype=np.float64)
    omega = omega_from_kappa(kappa_values, params.plate)
    mu = mu_resonator(omega, params.resonator)

    n_values = np.arange(
        -params.lattice.n_terms,
        params.lattice.n_terms + 1,
        dtype=np.float64,
    )

    kappa_col = kappa_values[:, None]
    kx = kappa_values * np.cos(psi)
    ky = kappa_values * np.sin(psi)
    kxn = kx[:, None] + 2.0 * np.pi * n_values[None, :] / params.lattice.a

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = kxn / kappa_col
        ratio2 = ratio * ratio

        lam = np.empty_like(ratio, dtype=np.complex128)
        propagating = np.abs(ratio) < 1.0
        lam[propagating] = -1j * np.sqrt(1.0 - ratio2[propagating])
        lam[~propagating] = np.sqrt(ratio2[~propagating] - 1.0) + 0.0j

        gam = np.sqrt(ratio2 + 1.0).astype(np.complex128)

        valid = (np.abs(lam) >= eps) & (np.abs(gam) >= eps)

        inv_term = np.zeros_like(lam, dtype=np.complex128)
        inv_term[valid] = 1.0 / lam[valid] - 1.0 / gam[valid]

        norm = 4.0 * params.lattice.a * kappa_values**3
        S = np.sum(inv_term, axis=1) / norm

        dy_abs = abs(delta_y)
        cross_term = np.zeros_like(lam, dtype=np.complex128)
        cross_term[valid] = (
            np.exp(-kappa_col[valid.any(axis=1)] * 0) if False else 0
        )

        cross_term[valid] = (
            np.exp(-kappa_col.repeat(lam.shape[1], axis=1)[valid] * lam[valid] * dy_abs) / lam[valid]
            - np.exp(-kappa_col.repeat(gam.shape[1], axis=1)[valid] * gam[valid] * dy_abs) / gam[valid]
        )

        cross_sum = np.sum(cross_term, axis=1) / norm
        phase = np.exp(1j * kx * delta_x)
        S1 = np.conj(phase) * cross_sum
        S2 = phase * cross_sum

        exp1 = 1.0 + 0.0j
        exp2 = np.exp(1j * ky * delta_y)

        denominator = (1.0 - mu * S) ** 2 - mu**2 * S1 * S2
        small = np.abs(denominator) < eps
        if np.any(small):
            denominator = denominator.copy()
            denominator[small] = eps + 0.0j

        zeta1 = ((1.0 - mu * S) * exp1 + mu * S1 * exp2) / denominator
        zeta2 = ((1.0 - mu * S) * exp2 + mu * S2 * exp1) / denominator

        sin_psi = np.sin(psi)
        amplitude = zeta1 + zeta2 * np.exp(1j * ky * delta_y)
        R0 = 1j * mu * amplitude / (4.0 * params.lattice.a * kappa_values**3 * sin_psi)
        R_raw = np.abs(R0) ** 2

    R_raw = np.nan_to_num(R_raw, nan=1.0, posinf=1.0, neginf=0.0)
    R = np.clip(R_raw, 0.0, 1.0)
    return omega, R


def generate_dataset(
    num_samples: int,
    params: ModelParams,
    kappa_min: float,
    kappa_max: float,
    psi: float,
    delta_x: float,
    delta_y: float,
    logit_eps: float,
    chunk_size: int,
) -> pd.DataFrame:
    if kappa_min <= 0.0:
        raise ValueError("kappa_min must be positive")
    if kappa_max <= kappa_min:
        raise ValueError("kappa_max must be greater than kappa_min")

    kappa_grid = np.linspace(kappa_min, kappa_max, num_samples, dtype=np.float64)
    parts: list[pd.DataFrame] = []

    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        kappa_chunk = kappa_grid[start:stop]
        omega_chunk, R_chunk = compute_R_for_kappa_grid(
            kappa_values=kappa_chunk,
            params=params,
            psi=psi,
            delta_x=delta_x,
            delta_y=delta_y,
            eps=1.0e-12,
        )
        parts.append(
            pd.DataFrame(
                {
                    "kappa": kappa_chunk,
                    "omega": omega_chunk,
                    "psi": psi,
                    "delta_x": delta_x,
                    "delta_y": delta_y,
                    "R": R_chunk,
                    "logit_R": make_logit_R(R_chunk, eps=logit_eps),
                }
            )
        )

    df = pd.concat(parts, ignore_index=True)
    if len(df) != num_samples:
        raise RuntimeError(f"expected {num_samples} rows, got {len(df)} rows.")
    if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("generated dataset contains non-finite values")
    return df


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def thousand_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 1000 != 0:
        raise argparse.ArgumentTypeError("num_samples must be a multiple of 1000")
    return parsed


def default_output_path(num_samples: int) -> Path:
    return Path("storage") / f"dataset_{num_samples // 1000}k.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed-geometry data for kappa -> R training.")

    parser.add_argument("--num_samples", type=thousand_count, default=10_000)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--kappa_min", type=float, default=1.0e-4)
    parser.add_argument("--kappa_max", type=float, default=20.0)

    parser.add_argument("--rho", type=float, default=PlateParams.rho)
    parser.add_argument("--thickness", type=float, default=PlateParams.h)
    parser.add_argument("--young_modulus", type=float, default=PlateParams.E)
    parser.add_argument("--poisson_ratio", type=float, default=PlateParams.nu)

    parser.add_argument("--period", type=float, default=LatticeParams.a)
    parser.add_argument("--n_terms", type=positive_int, default=LatticeParams.n_terms)

    parser.add_argument("--omega0", type=float, default=ResonatorParams.omega0)
    parser.add_argument("--damping_ratio", type=float, default=ResonatorParams.damping_ratio)
    parser.add_argument("--mu_scale", type=float, default=ResonatorParams.mu_scale)

    parser.add_argument("--geometry", choices=["zigzag", "half_period", "custom"], default="zigzag")
    parser.add_argument("--fixed_psi", type=float, default=np.pi / 6.0)
    parser.add_argument("--fixed_dx_fraction", type=float, default=0.5)
    parser.add_argument("--fixed_dy_fraction", type=float, default=0.5 * np.tan(np.pi / 6.0))

    parser.add_argument("--logit_eps", type=float, default=1.0e-6)
    parser.add_argument("--chunk_size", type=positive_int, default=5000)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    params = ModelParams(
        plate=PlateParams(
            rho=args.rho,
            h=args.thickness,
            E=args.young_modulus,
            nu=args.poisson_ratio,
        ),
        resonator=ResonatorParams(
            omega0=args.omega0,
            damping_ratio=args.damping_ratio,
            mu_scale=args.mu_scale,
        ),
        lattice=LatticeParams(
            a=args.period,
            n_terms=args.n_terms,
        ),
    )

    delta_x, delta_y = resolve_geometry(
        geometry=args.geometry,
        period=params.lattice.a,
        psi=args.fixed_psi,
        dx_fraction=args.fixed_dx_fraction,
        dy_fraction=args.fixed_dy_fraction,
    )

    output_path = Path(args.output) if args.output is not None else default_output_path(args.num_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        num_samples=args.num_samples,
        params=params,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        psi=args.fixed_psi,
        delta_x=delta_x,
        delta_y=delta_y,
        logit_eps=args.logit_eps,
        chunk_size=args.chunk_size,
    )
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"kappa: min={df['kappa'].min():.6e}, max={df['kappa'].max():.6e}")
    print(
        "R: "
        f"min={df['R'].min():.6e}, "
        f"max={df['R'].max():.6e}, "
        f"mean={df['R'].mean():.6e}, "
        f"median={df['R'].median():.6e}"
    )


if __name__ == "__main__":
    main()
