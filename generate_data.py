from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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
class DatasetRanges:
    omega_min: float = 2.0 * np.pi * 50.0
    omega_max: float = 2.0 * np.pi * 200.0
    psi_min: float = 0.05
    psi_max: float = np.pi - 0.05
    dx_min: float = 0.0
    dx_max: float = 0.05
    dy_min: float = 1.0e-4
    dy_max: float = 0.05


@dataclass(frozen=True)
class NumericalParams:
    eps: float = 1.0e-12
    max_attempt_multiplier: int = 80
    min_R: float = 0.0
    max_R: float = 1.0
    min_T: float = 0.0
    max_T: float = 2.0
    min_abs_denominator: float = 1.0e-6
    max_abs_zeta: float = 1.0e4


@dataclass(frozen=True)
class ModelParams:
    plate: PlateParams = PlateParams()
    resonator: ResonatorParams = ResonatorParams()
    lattice: LatticeParams = LatticeParams()
    ranges: DatasetRanges = DatasetRanges()
    numerical: NumericalParams = NumericalParams()


def kappa(omega: float, plate: PlateParams) -> float:
    return float((plate.rho * plate.h * omega**2 / plate.D) ** 0.25)


def mu_resonator(omega: float, resonator: ResonatorParams) -> complex:
    denominator = resonator.omega0**2 - omega**2 - 1j * resonator.gamma * omega
    return -resonator.mu_scale * omega**2 / denominator


def wavevector_components(psi: float, kappa_value: float) -> tuple[float, float]:
    return float(kappa_value * np.cos(psi)), float(kappa_value * np.sin(psi))


def lambda_gamma_vectorized(kxn: np.ndarray, kappa_value: float) -> tuple[np.ndarray, np.ndarray]:
    ratio = kxn / kappa_value
    ratio2 = ratio * ratio

    lam = np.empty_like(ratio, dtype=np.complex128)
    propagating = np.abs(ratio) < 1.0

    lam[propagating] = -1j * np.sqrt(1.0 - ratio2[propagating])
    lam[~propagating] = np.sqrt(ratio2[~propagating] - 1.0) + 0.0j

    gam = np.sqrt(ratio2 + 1.0).astype(np.complex128)
    return lam, gam


def lattice_sums(
    psi: float,
    delta_x: float,
    delta_y: float,
    kappa_value: float,
    lattice: LatticeParams,
    eps: float,
    n_values: np.ndarray,
) -> tuple[complex, complex, complex]:
    kx, _ = wavevector_components(psi, kappa_value)
    kxn = kx + 2.0 * np.pi * n_values / lattice.a

    lam, gam = lambda_gamma_vectorized(kxn, kappa_value)
    valid = (np.abs(lam) >= eps) & (np.abs(gam) >= eps)

    inv_term = np.zeros_like(lam, dtype=np.complex128)
    inv_term[valid] = 1.0 / lam[valid] - 1.0 / gam[valid]

    norm = 4.0 * lattice.a * kappa_value**3
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


def solve_zeta(
    psi: float,
    delta_y: float,
    mu: complex,
    S: complex,
    S1: complex,
    S2: complex,
    kappa_value: float,
    eps: float,
) -> tuple[complex, complex, complex]:
    _, ky = wavevector_components(psi, kappa_value)

    exp1 = 1.0 + 0.0j
    exp2 = np.exp(1j * ky * delta_y)

    denominator = (1.0 - mu * S) ** 2 - mu**2 * S1 * S2
    if abs(denominator) < eps:
        raise FloatingPointError("near-singular zeta system")

    zeta1 = ((1.0 - mu * S) * exp1 + mu * S1 * exp2) / denominator
    zeta2 = ((1.0 - mu * S) * exp2 + mu * S2 * exp1) / denominator

    return zeta1, zeta2, denominator


def reflection_transmission(
    psi: float,
    delta_y: float,
    mu: complex,
    zeta1: complex,
    zeta2: complex,
    kappa_value: float,
    lattice: LatticeParams,
    eps: float,
) -> tuple[float, float, complex]:
    _, ky = wavevector_components(psi, kappa_value)
    sin_psi = np.sin(psi)

    if abs(sin_psi) < eps:
        raise FloatingPointError("sin(psi) is too close to zero")

    amplitude = zeta1 + zeta2 * np.exp(1j * ky * delta_y)
    R0 = 1j * mu * amplitude / (4.0 * lattice.a * kappa_value**3 * sin_psi)

    R = float(abs(R0) ** 2)
    T = float(abs(1.0 + R0) ** 2)
    return R, T, R0


def physics_residuals(
    psi: float,
    delta_y: float,
    mu: complex,
    S: complex,
    S1: complex,
    S2: complex,
    zeta1: complex,
    zeta2: complex,
    kappa_value: float,
) -> tuple[float, float]:
    _, ky = wavevector_components(psi, kappa_value)

    exp1 = 1.0 + 0.0j
    exp2 = np.exp(1j * ky * delta_y)

    res1 = zeta1 - exp1 - mu * S * zeta1 - mu * S1 * zeta2
    res2 = zeta2 - exp2 - mu * S2 * zeta1 - mu * S * zeta2

    return float(abs(res1)), float(abs(res2))


def sample_parameters(rng: np.random.Generator, ranges: DatasetRanges) -> tuple[float, float, float, float]:
    omega = rng.uniform(ranges.omega_min, ranges.omega_max)
    psi = rng.uniform(ranges.psi_min, ranges.psi_max)
    delta_x = rng.uniform(ranges.dx_min, ranges.dx_max)
    delta_y = rng.uniform(ranges.dy_min, ranges.dy_max)
    return float(omega), float(psi), float(delta_x), float(delta_y)


def compute_sample(
    omega: float,
    psi: float,
    delta_x: float,
    delta_y: float,
    params: ModelParams,
    n_values: np.ndarray,
) -> dict[str, float]:
    kap = kappa(omega, params.plate)
    mu = mu_resonator(omega, params.resonator)

    S, S1, S2 = lattice_sums(
        psi=psi,
        delta_x=delta_x,
        delta_y=delta_y,
        kappa_value=kap,
        lattice=params.lattice,
        eps=params.numerical.eps,
        n_values=n_values,
    )

    zeta1, zeta2, denominator = solve_zeta(
        psi=psi,
        delta_y=delta_y,
        mu=mu,
        S=S,
        S1=S1,
        S2=S2,
        kappa_value=kap,
        eps=params.numerical.eps,
    )

    if abs(denominator) < params.numerical.min_abs_denominator:
        raise FloatingPointError("zeta system is poorly conditioned")

    if max(abs(zeta1), abs(zeta2)) > params.numerical.max_abs_zeta:
        raise FloatingPointError("zeta is too large")

    R, T, R0 = reflection_transmission(
        psi=psi,
        delta_y=delta_y,
        mu=mu,
        zeta1=zeta1,
        zeta2=zeta2,
        kappa_value=kap,
        lattice=params.lattice,
        eps=params.numerical.eps,
    )

    if not (params.numerical.min_R <= R <= params.numerical.max_R):
        raise FloatingPointError("R is outside the physical range")

    if not (params.numerical.min_T <= T <= params.numerical.max_T):
        raise FloatingPointError("T is outside the allowed range")

    res1, res2 = physics_residuals(
        psi=psi,
        delta_y=delta_y,
        mu=mu,
        S=S,
        S1=S1,
        S2=S2,
        zeta1=zeta1,
        zeta2=zeta2,
        kappa_value=kap,
    )

    row = {
        "omega": omega,
        "psi": psi,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "kappa": kap,
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
        "T": T,
        "Re_R0": R0.real,
        "Im_R0": R0.imag,
        "energy_balance": R + T,
        "abs_denominator": abs(denominator),
        "residual_1": res1,
        "residual_2": res2,
    }

    if not np.all(np.isfinite(list(row.values()))):
        raise FloatingPointError("non-finite value in generated row")

    return row


def generate_dataset(num_samples: int, params: ModelParams, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_values = np.arange(-params.lattice.n_terms, params.lattice.n_terms + 1, dtype=np.float64)

    rows: list[dict[str, float]] = []
    attempts = 0
    max_attempts = num_samples * params.numerical.max_attempt_multiplier

    with tqdm(total=num_samples, desc="Generating", unit="row") as pbar:
        while len(rows) < num_samples and attempts < max_attempts:
            attempts += 1
            omega, psi, delta_x, delta_y = sample_parameters(rng, params.ranges)

            try:
                row = compute_sample(omega, psi, delta_x, delta_y, params, n_values)
            except FloatingPointError:
                continue

            rows.append(row)
            pbar.update(1)

    if len(rows) < num_samples:
        raise RuntimeError(
            f"Attempts: {attempts}. Increase max_attempt_multiplier or relax constraints."
        )

    df = pd.DataFrame(rows)
    df.attrs["attempts"] = attempts
    df.attrs["accepted"] = len(rows)
    df.attrs["acceptance_rate"] = len(rows) / attempts
    return df


def print_diagnostics(df: pd.DataFrame, attempts: int) -> None:
    acceptance_rate = len(df) / attempts

    print("\nDataset diagnostics")
    print("-------------------")
    print(f"rows:            {len(df)}")
    print(f"attempts:        {attempts}")
    print(f"acceptance rate: {acceptance_rate:.2%}")
    print(
        "R: "
        f"min={df['R'].min():.6e}, "
        f"mean={df['R'].mean():.6e}, "
        f"median={df['R'].median():.6e}, "
        f"max={df['R'].max():.6e}"
    )


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
    parser = argparse.ArgumentParser(description="Generate a clean synthetic dataset for PINN training.")

    parser.add_argument("--num_samples", type=thousand_count, default=30_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rho", type=float, default=PlateParams.rho)
    parser.add_argument("--thickness", type=float, default=PlateParams.h)
    parser.add_argument("--young_modulus", type=float, default=PlateParams.E)
    parser.add_argument("--poisson_ratio", type=float, default=PlateParams.nu)

    parser.add_argument("--omega_min", type=float, default=DatasetRanges.omega_min)
    parser.add_argument("--omega_max", type=float, default=DatasetRanges.omega_max)
    parser.add_argument("--psi_min", type=float, default=DatasetRanges.psi_min)
    parser.add_argument("--psi_max", type=float, default=DatasetRanges.psi_max)
    parser.add_argument("--dx_min", type=float, default=DatasetRanges.dx_min)
    parser.add_argument("--dx_max", type=float, default=DatasetRanges.dx_max)
    parser.add_argument("--dy_min", type=float, default=DatasetRanges.dy_min)
    parser.add_argument("--dy_max", type=float, default=DatasetRanges.dy_max)

    parser.add_argument("--period", type=float, default=LatticeParams.a)
    parser.add_argument("--n_terms", type=positive_int, default=LatticeParams.n_terms)

    parser.add_argument("--omega0", type=float, default=ResonatorParams.omega0)
    parser.add_argument("--damping_ratio", type=float, default=ResonatorParams.damping_ratio)
    parser.add_argument("--mu_scale", type=float, default=ResonatorParams.mu_scale)

    parser.add_argument("--max_R", type=float, default=NumericalParams.max_R)
    parser.add_argument("--max_T", type=float, default=NumericalParams.max_T)
    parser.add_argument("--min_abs_denominator", type=float, default=NumericalParams.min_abs_denominator)
    parser.add_argument("--max_abs_zeta", type=float, default=NumericalParams.max_abs_zeta)
    parser.add_argument("--max_attempt_multiplier", type=positive_int, default=NumericalParams.max_attempt_multiplier)

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
        ranges=DatasetRanges(
            omega_min=args.omega_min,
            omega_max=args.omega_max,
            psi_min=args.psi_min,
            psi_max=args.psi_max,
            dx_min=args.dx_min,
            dx_max=args.dx_max,
            dy_min=args.dy_min,
            dy_max=args.dy_max,
        ),
        numerical=NumericalParams(
            max_R=args.max_R,
            max_T=args.max_T,
            min_abs_denominator=args.min_abs_denominator,
            max_abs_zeta=args.max_abs_zeta,
            max_attempt_multiplier=args.max_attempt_multiplier,
        ),
    )

    output_path = Path(args.output) if args.output is not None else default_output_path(args.num_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(args.num_samples, params=params, seed=args.seed)
    df.to_csv(output_path, index=False)

    attempts = int(df.attrs["attempts"])

    print(f"\nSaved dataset to: {output_path}")
    print_diagnostics(df, attempts=attempts)


if __name__ == "__main__":
    main()
