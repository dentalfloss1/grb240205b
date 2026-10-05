#!/usr/bin/env python3
"""Fit radio data and plot radio fits plus held-out optical/IR extrapolations."""

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "grbfit-matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from grbfit.data import load_data, prepare_fit_data
from grbfit.fit import evaluate_model_components, make_model, run_mcmc, samples_to_physical
from grbfit.run import calculate_goodness_of_fit, normalize_config


TRIGGER = dt.datetime(2024, 2, 5, 22, 13, 8)
REPOSITORY_DIR = Path(__file__).resolve().parent.parent
PUBLICATION_PLOTS_DIR = Path(__file__).resolve().parent
LEGACY_DATA_DIR = REPOSITORY_DIR / "old"
FORWARD_REVERSE_DIR = REPOSITORY_DIR / "forwardreverse"
FORWARD_ONLY_DIR = REPOSITORY_DIR / "forwardonly"
FORWARD_REVERSE_CONFIGS = {
    "wind": FORWARD_REVERSE_DIR / "wind" / "config.yaml",
    "ism": FORWARD_REVERSE_DIR / "ism" / "config.yaml",
}
FORWARD_ONLY_CONFIGS = {
    "wind": FORWARD_ONLY_DIR / "wind" / "config.yaml",
    "ism": FORWARD_ONLY_DIR / "ism" / "config.yaml",
}

RADIO_PANELS = [
    ("0.8-1.6 GHz", lambda freq: freq < 2.0),
    ("2.6-3.1 GHz", lambda freq: (freq >= 2.0) & (freq < 4.0)),
    ("5.5 GHz", lambda freq: np.isclose(freq, 5.5)),
    ("9 GHz", lambda freq: np.isclose(freq, 9.0)),
    ("16.7 GHz", lambda freq: np.isclose(freq, 16.7)),
    ("21.2 GHz", lambda freq: np.isclose(freq, 21.2)),
]

PANEL_SPECS = RADIO_PANELS

# Detection markers only; v/^ are reserved for upper/lower limits.
MARKERS = ["o", "s", "X", "*"]
# Okabe-Ito colors, supplemented by marker shapes and line styles.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "black"]
PROFILE_STYLES = {
    "wind": ("#0072B2", "-", "Wind: forward + reverse"),
    "ism": ("#D55E00", "--", "ISM: forward + reverse"),
}

BAND_LABELS = {
    0.81: "0.81 GHz",
    1.3: "1.3 GHz",
    1.6: "1.6 GHz",
    2.6: "2.6 GHz",
    3.1: "3.1 GHz",
    5.5: "5.5 GHz",
    9.0: "9 GHz",
    16.7: "16.7 GHz",
    21.2: "21.2 GHz",
}
OPTICAL_IR_BANDS = {
    180380.540313: "H",
    239833.9664: "J",
    371950.940447: "I",
    455611.638298: "R",
    485415.249352: "r",
    544087.945554: "V",
    628495.719078: "g",
    673690.916854: "B",
}
EXTRAPOLATION_REFERENCE_GHZ = 485415.249352  # r band, observer frame
EXTRAPOLATION_P = 2.2
RADIO_BAND_STYLES = {
    freq: (MARKERS[index % len(MARKERS)], COLORS[index % len(COLORS)])
    for _, select in RADIO_PANELS
    for index, freq in enumerate(freq for freq in BAND_LABELS if select(freq))
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit radio data with grbfit and plot radio fits and optical/IR extrapolations."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Use one forward-reverse config for all requested profiles; normally leave unset.",
    )
    parser.add_argument("--wind-config", default=str(FORWARD_REVERSE_CONFIGS["wind"]))
    parser.add_argument("--ism-config", default=str(FORWARD_REVERSE_CONFIGS["ism"]))
    parser.add_argument(
        "--forward-only-config",
        default=None,
        help="Use one forward-only config for all requested profiles; normally leave unset.",
    )
    parser.add_argument(
        "--forward-only-wind-config",
        default=str(FORWARD_ONLY_CONFIGS["wind"]),
    )
    parser.add_argument(
        "--forward-only-ism-config",
        default=str(FORWARD_ONLY_CONFIGS["ism"]),
    )
    parser.add_argument(
        "--sample-cache",
        default=str(PUBLICATION_PLOTS_DIR / "publication_samples_{profile}.npz"),
    )
    parser.add_argument(
        "--forward-only-sample-cache",
        default=str(
            PUBLICATION_PLOTS_DIR / "publication_forward_only_samples_{profile}.npz"
        ),
    )
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--extrapolation-draws", type=int, default=500)
    parser.add_argument(
        "--extrapolation-output",
        default=str(PUBLICATION_PLOTS_DIR / "publication_optical_ir_extrapolation.png"),
        help="Single-panel multiband extrapolation figure; also writes a matching PDF.",
    )
    parser.add_argument(
        "--output",
        default=str(PUBLICATION_PLOTS_DIR / "publication_lightcurves_{profile}.png"),
    )
    parser.add_argument(
        "--residual-output",
        default=str(PUBLICATION_PLOTS_DIR / "publication_residuals_{profile}.png"),
    )
    parser.add_argument(
        "--fit-statistics-output",
        default=str(PUBLICATION_PLOTS_DIR / "publication_fit_statistics.csv"),
        help="CSV containing radio-only fit statistics for every fitted model.",
    )
    parser.add_argument("--seed", type=int, default=240205)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["wind", "ism"],
        default=["wind", "ism"],
        help="Profiles to fit and plot. Defaults to both wind and ISM.",
    )
    parser.add_argument(
        "--mcmc-mode",
        choices=["adaptive", "fixed"],
        default=None,
        help="Override fit.mcmc_mode from config.yaml.",
    )
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--nsteps", type=int, default=None)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--forward-only-mcmc-mode",
        choices=["adaptive", "fixed"],
        default=None,
        help="Override fit.mcmc_mode only for the forward-only comparison fit.",
    )
    parser.add_argument("--forward-only-burn-in", type=int, default=None)
    parser.add_argument("--forward-only-nsteps", type=int, default=None)
    parser.add_argument("--forward-only-nwalkers", type=int, default=None)
    parser.add_argument("--forward-only-max-steps", type=int, default=None)
    return parser.parse_args()


def resolve_profile_config(args, profile):
    if args.config is not None:
        return Path(args.config).expanduser().resolve()
    if profile == "wind":
        return Path(args.wind_config).expanduser().resolve()
    return Path(args.ism_config).expanduser().resolve()


def resolve_forward_only_profile_config(args, profile):
    if args.forward_only_config is not None:
        return Path(args.forward_only_config).expanduser().resolve()
    if profile == "wind":
        return Path(args.forward_only_wind_config).expanduser().resolve()
    return Path(args.forward_only_ism_config).expanduser().resolve()


def absolutize_data_paths(cfg, config_path):
    config_dir = Path(config_path).resolve().parent
    for key in ("radio_file", "other_file", "batxrt_file"):
        value = cfg.get("data", {}).get(key)
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        cfg["data"][key] = str(path)


def load_config(path):
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    absolutize_data_paths(cfg, path)
    cfg = normalize_config(cfg)
    # Non-radio observations must never enter the publication fits.
    cfg["data"].pop("other_file", None)
    cfg["data"].pop("batxrt_file", None)
    return cfg


def radio_data(df):
    """Enforce radio-only inputs independently of frequency cuts in a config."""
    return df.loc[df["instrument"] == "radio"].copy()


def load_comparison_data(config_path):
    """Load held-out photometry separately from the fitting configuration."""
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    absolutize_data_paths(cfg, config_path)
    path = cfg.get("data", {}).get("other_file")
    if path is None:
        return pd.DataFrame(columns=["obsdate", "freq", "flux", "err", "rms"])
    return pd.read_csv(path)


def config_text(path):
    return Path(path).read_text()


def mcmc_overrides(args):
    return {
        key: value
        for key, value in {
            "mcmc_mode": args.mcmc_mode,
            "burn_in": args.burn_in,
            "nsteps": args.nsteps,
            "nwalkers": args.nwalkers,
            "max_steps": args.max_steps,
        }.items()
        if value is not None
    }


def forward_only_mcmc_overrides(args):
    return {
        key: value
        for key, value in {
            "mcmc_mode": args.forward_only_mcmc_mode,
            "burn_in": args.forward_only_burn_in,
            "nsteps": args.forward_only_nsteps,
            "nwalkers": args.forward_only_nwalkers,
            "max_steps": args.forward_only_max_steps,
        }.items()
        if value is not None
    }


def apply_mcmc_overrides(cfg, overrides):
    cfg["fit"].update(overrides)


def cache_key(config_path, overrides, profile):
    return json.dumps(
        {
            "config_text": config_text(config_path),
            "mcmc_overrides": overrides,
            "profile": profile,
            "data_scope": "radio_only_v1",
            "radio_sha256": hashlib.sha256(
                Path(load_config(config_path)["data"]["radio_file"]).read_bytes()
            ).hexdigest(),
        },
        sort_keys=True,
    )


def cache_matches(cache_path, config_path, overrides, profile):
    if not Path(cache_path).exists():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if "cache_key" in cache:
                return str(cache["cache_key"]) == cache_key(config_path, overrides, profile)
            return False
    except Exception:
        return False


def get_samples(cfg, config_path, cache_path, overrides, profile, force_refit=False):
    if not force_refit and cache_matches(cache_path, config_path, overrides, profile):
        with np.load(cache_path, allow_pickle=False) as cache:
            samples = cache["samples"]
            keys = [str(k) for k in cache["keys"]]
            diagnostics = json.loads(str(cache["diagnostics_json"]))
            fixed_params = json.loads(str(cache["fixed_params_json"]))
        cfg["fit"]["param_keys"] = keys
        cfg["fit"]["fixed_params"] = fixed_params
        print(f"Loaded posterior samples from {cache_path}")
        return keys, samples, diagnostics

    df = radio_data(load_data(cfg))
    xdata, ydata, yerr, upper, excluded, _ = prepare_fit_data(df, cfg)
    print(
        f"Radio input: {len(df)} rows; likelihood: {len(ydata)} detections; "
        f"limits: {len(upper)}; outside fitting cuts: {len(excluded)}"
    )
    keys, sampler = run_mcmc(cfg, xdata, ydata, yerr)
    diagnostics = getattr(sampler, "grbfit_diagnostics", {})
    thin = int(diagnostics.get("thin", 1))
    flat_sampling = sampler.get_chain(discard=0, thin=thin, flat=True)
    samples = samples_to_physical(flat_sampling, keys)

    np.savez_compressed(
        cache_path,
        samples=samples,
        keys=np.array(keys),
        diagnostics_json=json.dumps(diagnostics),
        fixed_params_json=json.dumps(cfg["fit"].get("fixed_params", {})),
        config_text=config_text(config_path),
        cache_key=cache_key(config_path, overrides, profile),
    )
    print(f"Saved posterior samples to {cache_path}")
    return keys, samples, diagnostics


def freq_label(freq):
    for known, label in OPTICAL_IR_BANDS.items():
        if np.isclose(freq, known, rtol=0, atol=max(1e-6, abs(known) * 1e-9)):
            return label
    for known, label in BAND_LABELS.items():
        if np.isclose(freq, known, rtol=0, atol=max(1e-6, abs(known) * 1e-9)):
            return label
    if freq > 1e3:
        wavelength_um = 299792458.0 / (freq * 1e9) * 1e6
        return f"{wavelength_um:.2f} um"
    return f"{freq:g} GHz"


def panel_mask(df, panel_index):
    if "instrument" in df.columns:
        instrument_mask = df["instrument"] == "radio"
    else:
        instrument_mask = np.ones(len(df), dtype=bool)
    return instrument_mask & PANEL_SPECS[panel_index][1](df["freq"].to_numpy())


def panel_time_limits(panel_index):
    return (1e-2, 365)


def positive(values):
    values = np.asarray(values)
    return np.isfinite(values) & (values > 0)


def total_error(rows):
    return np.sqrt(rows["err"].to_numpy() ** 2 + rows["rms"].to_numpy() ** 2)


def is_nine_ghz_third_early_point(rows):
    ordered = rows[np.isclose(rows["freq"], 9.0)].sort_values("obsdate")
    early = ordered[ordered["obsdate"] < 1.0]
    if len(early) < 3:
        return pd.Series(False, index=rows.index)
    target_index = early.index[2]
    return rows.index == target_index


def plot_data(ax, subset):
    handles = []
    unique_freqs = np.sort(subset["freq"].unique())
    for i, freq in enumerate(unique_freqs):
        rows = subset[np.isclose(subset["freq"], freq)]
        marker, color = RADIO_BAND_STYLES.get(
            float(freq), (MARKERS[i % len(MARKERS)], COLORS[i % len(COLORS)])
        )
        label = freq_label(freq)

        force_detection = is_nine_ghz_third_early_point(rows)
        det = rows[((rows["flux"] > 0) & (rows["err"] > 0)) | force_detection].copy()
        if len(det) > 0:
            early_9 = np.isclose(freq, 9.0) & (det["obsdate"] < 1.0)
            for open_marker in [False, True]:
                subdet = det[early_9 == open_marker]
                if len(subdet) == 0:
                    continue
                handle = ax.errorbar(
                    subdet["obsdate"],
                    subdet["flux"],
                    yerr=total_error(subdet),
                    fmt=marker,
                    linestyle="none",
                    color=color,
                    markerfacecolor="none" if open_marker else color,
                    markeredgecolor=color,
                    markersize=4.5,
                    elinewidth=0.9,
                    capsize=0,
                    label=label if not handles and len(unique_freqs) == 1 else label,
                )
                handles.append(handle)

        limits = rows[~(((rows["flux"] > 0) & (rows["err"] > 0)) | force_detection)]
        if len(limits) > 0:
            ylimit = 3.0 * np.abs(limits["rms"].to_numpy())
            ok = positive(ylimit)
            if np.any(ok):
                ax.scatter(
                    limits["obsdate"].to_numpy()[ok],
                    ylimit[ok],
                    marker="v",
                    color=color,
                    s=28,
                    alpha=0.8,
                    label=f"{label} 3$\\sigma$ limit",
                )
    return handles


def load_root_observation_data():
    path = LEGACY_DATA_DIR / "grbmeas.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    return add_observation_dates(data)


def load_check_source_data():
    path = LEGACY_DATA_DIR / "checksrc.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    return add_observation_dates(data)


def parse_observation_time(value):
    return pd.to_datetime(value).to_pydatetime()


def add_observation_dates(data):
    data = data.copy()
    starts = [parse_observation_time(value) for value in data["start"]]
    stops = [parse_observation_time(value) for value in data["stop"]]
    startdate = [(value - TRIGGER).total_seconds() / 86400 for value in starts]
    stopdate = [(value - TRIGGER).total_seconds() / 86400 for value in stops]
    data["startdate"] = np.minimum(startdate, stopdate)
    data["stopdate"] = np.maximum(startdate, stopdate)
    data["obsdate"] = 0.5 * (data["startdate"] + data["stopdate"])
    return data


def shade_observation_spans(ax, panel_index, root_obs):
    if root_obs is None or len(root_obs) == 0 or panel_index >= len(RADIO_PANELS):
        return
    subset = root_obs[panel_mask(root_obs, panel_index)]
    for obs in np.sort(subset["obs"].unique()):
        rows = subset[subset["obs"] == obs]
        start = rows["startdate"].min()
        stop = rows["stopdate"].max()
        if np.isfinite(start) and np.isfinite(stop) and stop > 0:
            ax.axvspan(max(start, 1e-4), stop, alpha=0.15, color="gray", zorder=0)


def plot_check_source(ax, panel_index, check_source):
    if check_source is None or len(check_source) == 0 or panel_index != 2:
        return
    rows = check_source[check_source["band"] == "C"]
    if len(rows) == 0:
        return
    ax.errorbar(
        rows["obsdate"],
        rows["flux"],
        yerr=total_error(rows),
        fmt="x",
        color="#009E73",
        markersize=4,
        elinewidth=0.9,
        linestyle="none",
        label="check source",
    )


def deduplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen or label.startswith("_"):
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, fontsize=7, loc="best")


def model_time_grid(subset):
    times = subset["obsdate"].to_numpy()
    times = times[positive(times)]
    if len(times) == 0:
        return np.geomspace(1e-2, 365, 250)
    tmin = max(1e-3, min(1e-2, times.min() * 0.7))
    tmax = max(365.0, times.max() * 1.4)
    return np.geomspace(tmin, tmax, 300)


def representative_frequency(subset):
    freqs = np.sort(subset["freq"].unique())
    if len(freqs) == 0:
        return None
    return freqs[-1]


def draw_models(ax, cfg, samples, subset, n_draws, rng, forward_only_cfg=None, forward_only_samples=None):
    model = make_model(cfg)
    median_theta = np.median(samples, axis=0)
    draw_indices = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)

    freq = representative_frequency(subset)
    if freq is None:
        return
    t_line = model_time_grid(subset)
    nu_line = np.full_like(t_line, freq, dtype=float)

    for index in draw_indices:
        y = model(samples[index], (t_line, nu_line)) * 1e6
        ok = positive(y)
        ax.plot(t_line[ok], y[ok], color="#0072B2", alpha=0.14, linewidth=0.9)

    components = evaluate_model_components(cfg, median_theta, (t_line, nu_line))
    total = components["total"] * 1e6
    ok = positive(total)
    ax.plot(
        t_line[ok],
        total[ok],
        color="#0072B2",
        linewidth=1.9,
        label=f"{freq_label(freq)} Forward+Reverse",
    )

    if forward_only_cfg is not None and forward_only_samples is not None and len(forward_only_samples) > 0:
        forward_only_model = make_model(forward_only_cfg)
        forward_only_theta = np.median(forward_only_samples, axis=0)
        forward_only = forward_only_model(forward_only_theta, (t_line, nu_line)) * 1e6
        ok = positive(forward_only)
        ax.plot(
            t_line[ok],
            forward_only[ok],
            color="0.45",
            linewidth=1.4,
            linestyle="--",
            label=f"{freq_label(freq)} Forward Only Fit",
        )

    forward = components["forward"] * 1e6
    ok = positive(forward)
    ax.plot(t_line[ok], forward[ok], color="#D55E00", alpha=0.55, linewidth=1.1, linestyle="-.")

    reverse = components["reverse"] * 1e6
    ok = positive(reverse)
    if np.any(ok):
        ax.plot(t_line[ok], reverse[ok], color="#D55E00", alpha=0.55, linewidth=1.1, linestyle=":")


def set_lightcurve_limits(ax, subset):
    vals = []
    det = subset[subset["flux"] > 0]
    if len(det) > 0:
        vals.extend(det["flux"].to_numpy())
    limits = subset[~((subset["flux"] > 0) & (subset["err"] > 0))]
    if len(limits) > 0:
        vals.extend(3.0 * np.abs(limits["rms"].to_numpy()))
    vals = np.asarray(vals)
    vals = vals[positive(vals)]
    if len(vals) == 0:
        ax.set_ylim(1, 1e4)
        return
    ymin = vals.min() * 0.35
    ymax = vals.max() * 2.5
    if ymax / ymin < 100:
        center = np.sqrt(ymin * ymax)
        ymin = center / 10
        ymax = center * 10
    ax.set_ylim(ymin, ymax)


def make_lightcurve_plot(
    cfg,
    samples,
    df,
    output,
    n_draws,
    rng,
    profile_title,
    root_obs,
    check_source,
    forward_only_cfg=None,
    forward_only_samples=None,
):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex="row")
    axes = axes.flatten()

    for panel_index, ax in enumerate(axes):
        subset = df[panel_mask(df, panel_index)].copy()
        if len(subset) == 0:
            ax.set_visible(False)
            continue

        shade_observation_spans(ax, panel_index, root_obs)
        plot_data(ax, subset)
        plot_check_source(ax, panel_index, check_source)
        draw_models(
            ax,
            cfg,
            samples,
            subset,
            n_draws,
            rng,
            forward_only_cfg=forward_only_cfg,
            forward_only_samples=forward_only_samples,
        )

        ax.set_title(PANEL_SPECS[panel_index][0], fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*panel_time_limits(panel_index))
        set_lightcurve_limits(ax, subset)
        ax.set_ylabel(r"Flux Density ($\mu$Jy)")
        deduplicate_legend(ax)

    for ax in axes[-2:]:
        ax.set_xlabel("Days post-trigger")

    fig.suptitle(profile_title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=250)
    plt.close(fig)
    print(f"Saved {output}")


def extrapolation_summary(cfg, samples, times, freq, indices):
    """Observer-frame predictions in microJy; no adjustment to held-out data."""
    model = make_model(cfg)
    xdata = (times, np.full_like(times, freq, dtype=float))
    predictions = np.asarray([model(samples[index], xdata) * 1e6 for index in indices])
    if not np.all(positive(predictions)):
        raise ValueError(f"Nonpositive or nonfinite extrapolation at {freq:g} GHz")
    return np.percentile(predictions, [16, 50, 84], axis=0)


def scale_photometry_to_reference(observations, reference_freq=EXTRAPOLATION_REFERENCE_GHZ,
                                  p=EXTRAPOLATION_P):
    """Convert F_nu and its uncertainties to F_nu0 for F_nu proportional to nu^-beta."""
    scaled = observations.copy()
    factors = (scaled["freq"] / reference_freq) ** ((p - 1) / 2)
    for column in ("flux", "err", "rms"):
        scaled[column] = scaled[column] * factors
    return scaled


def make_extrapolation_plot(jobs, output, n_draws, seed):
    if n_draws < 1:
        raise ValueError("--extrapolation-draws must be positive")
    # Compare each model against the same observations, without duplicating rows
    # from the two profile configs (or discarding repeated measurements).
    observations = jobs[0]["comparison_df"].reset_index(drop=True)
    columns = ["obsdate", "freq", "flux", "err", "rms"]
    canonical = observations[columns].sort_values(columns).reset_index(drop=True)
    for job in jobs[1:]:
        other = job["comparison_df"][columns].sort_values(columns).reset_index(drop=True)
        if not canonical.equals(other):
            raise ValueError("Profile configs reference different optical/IR observations")
    if observations.empty:
        raise ValueError("No optical/IR observations available for extrapolation")
    if not np.all(positive(observations["obsdate"])):
        raise ValueError("Extrapolation observations require finite positive times")
    known = np.zeros(len(observations), dtype=bool)
    for freq in OPTICAL_IR_BANDS:
        known |= np.isclose(observations["freq"], freq, rtol=1e-9, atol=1e-6)
    if not known.all():
        raise ValueError("Unrecognized comparison band; extend OPTICAL_IR_BANDS before plotting")
    observations = scale_photometry_to_reference(observations)

    tmin = observations["obsdate"].min() / 1.5
    tmax = observations["obsdate"].max() * 1.5
    times = np.geomspace(tmin, tmax, 300)
    draws = {}
    for job in jobs:
        rng = np.random.default_rng(seed + (100 if job["profile"] == "wind" else 101))
        draws[job["profile"]] = rng.choice(
            len(job["samples"]), min(n_draws, len(job["samples"])), replace=False
        )

    # Redundant band encoding keeps the observations identifiable in grayscale.
    band_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                   "#E69F00", "#56B4E9", "#222222", "#8C6D00"]
    band_markers = ["o", "s", "D", "P", "X", "h", "*", "8"]
    with plt.rc_context({"font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.5, 5.25), layout="constrained")
        extents = []
        band_handles = []
        # Evaluate each physical model directly at the reference frequency.
        for job in jobs:
            color, linestyle, _ = PROFILE_STYLES[job["profile"]]
            low, median, high = extrapolation_summary(
                job["cfg"], job["samples"], times, EXTRAPOLATION_REFERENCE_GHZ,
                draws[job["profile"]]
            )
            ax.fill_between(times, low, high, color=color, alpha=0.13, linewidth=0)
            ax.plot(times, median, color=color, linestyle=linestyle,
                    linewidth=2, zorder=2)
            extents.extend([low.min(), high.max()])
        for (freq, band), color, marker in zip(
                OPTICAL_IR_BANDS.items(), band_colors, band_markers):
            rows = observations[np.isclose(observations["freq"], freq, rtol=1e-9, atol=1e-6)]
            detected = (rows["flux"] > 0) & (rows["err"] > 0)
            det = rows[detected]
            if len(det):
                errors = total_error(det)
                ax.errorbar(det["obsdate"], det["flux"], yerr=errors, fmt=marker,
                            color=color, markerfacecolor="white", markeredgewidth=1.3,
                            markersize=7 if marker == "*" else 5.5,
                            elinewidth=1.25, capsize=3.5, capthick=1.25,
                            barsabove=True, zorder=5)
                extents.extend(det["flux"] - errors)
                extents.extend(det["flux"] + errors)
            limits = rows[~detected]
            if len(limits):
                values = 3 * np.abs(limits["rms"].to_numpy())
                valid = positive(values)
                ax.scatter(limits["obsdate"].to_numpy()[valid], values[valid], marker="v",
                           color=color, s=35, zorder=5)
                extents.extend(values[valid])
            band_handles.append(Line2D([], [], color=color, marker=marker,
                                       markerfacecolor="white", markeredgewidth=1.3,
                                       markersize=6, linestyle="none", label=band))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(tmin, tmax)
        extents = np.asarray(extents)
        extents = extents[positive(extents)]
        ax.set_ylim(extents.min() / 1.5, extents.max() * 1.5)
        ax.set_xlabel("Days post-trigger")
        ax.set_ylabel(r"$r$-band-equivalent flux density ($\mu$Jy)")
        ax.set_title("Extrapolated Lightcurves", pad=10)
        ax.grid(which="major", color="0.92", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", direction="out")

        model_handles = [Line2D([], [], color=PROFILE_STYLES[job["profile"]][0], linewidth=2,
                               linestyle=PROFILE_STYLES[job["profile"]][1],
                               label="Wind medium" if job["profile"] == "wind" else "ISM")
                         for job in jobs]
        model_legend = ax.legend(handles=model_handles, loc="upper right",
                                 title="Radio-fitted forward + reverse",
                                 frameon=False, handlelength=3, fontsize=11, title_fontsize=10)
        ax.add_artist(model_legend)
        if ((observations["flux"] <= 0) | (observations["err"] <= 0)).any():
            band_handles.append(Line2D([], [], color="black", marker="v", linestyle="none",
                                       label=r"3$\sigma$ limit"))
        ax.legend(handles=band_handles, loc="lower left", ncol=4,
                  frameon=False, fontsize=11,
                  handletextpad=0.3, columnspacing=1.0, borderaxespad=0.7)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, facecolor="white")
        if output.suffix.lower() != ".pdf":
            fig.savefig(output.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        print(f"Saved {output} and matching PDF")


def residual_dataframe(cfg, samples, df):
    model = make_model(cfg)
    fit_df = radio_data(df)
    if cfg["burst"].get("z") is not None:
        fit_df["freq_rest"] = fit_df["freq"] * (1 + cfg["burst"]["z"])
    else:
        fit_df["freq_rest"] = fit_df["freq"]

    fitstart = cfg["burst"]["fitstart"]
    max_rest_freq = cfg["fit"].get("max_rest_freq", np.inf)
    fit_xrt = cfg["fit"].get("fit_xrt", False)
    if fit_xrt:
        fit_mask = fit_df["obsdate"] > fitstart
    else:
        fit_mask = (fit_df["obsdate"] > fitstart) & (fit_df["freq_rest"] < max_rest_freq)
    det_mask = (fit_df["flux"] > 0) & (fit_df["err"] > 0)
    fit_df = fit_df[fit_mask & det_mask].copy()

    if len(fit_df) == 0:
        fit_df["residual_sigma"] = []
        return fit_df

    theta = np.median(samples, axis=0)
    t = fit_df["obsdate"].to_numpy()
    nu = fit_df["freq"].to_numpy()
    y = fit_df["flux"].to_numpy() * 1e-6
    yerr = np.sqrt(fit_df["err"].to_numpy() ** 2 + fit_df["rms"].to_numpy() ** 2) * 1e-6
    ymodel = model(theta, (t, nu))

    valid = positive(y) & positive(yerr) & positive(ymodel)
    residual = np.full(len(fit_df), np.nan)
    log_residual = np.log10(y[valid]) - np.log10(ymodel[valid])
    log_sigma = yerr[valid] / (y[valid] * np.log(10))
    residual[valid] = log_residual / log_sigma
    fit_df["residual_sigma"] = residual
    return fit_df[np.isfinite(fit_df["residual_sigma"])]


def residual_axis_limit(residual_frames):
    max_abs = 3.0
    for resid in residual_frames:
        if len(resid) == 0:
            continue
        max_abs = max(max_abs, np.nanmax(np.abs(resid["residual_sigma"])) * 1.15)
    return max_abs


def fit_statistics(cfg, samples, df):
    """Calculate goodness-of-fit metrics from all fitted detections."""
    xdata, ydata, yerr, _, _, _ = prepare_fit_data(radio_data(df), cfg)
    return calculate_goodness_of_fit(cfg, samples, xdata, ydata, yerr)


def write_fit_statistics(rows, output):
    columns = [
        "profile",
        "model",
        "data",
        "ndata",
        "nfit",
        "DOF",
        "chisq",
        "redchisq",
        "AIC",
        "BIC",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    print(f"Saved radio-only fit statistics to {output}")


def make_residual_plot(cfg, samples, df, output, profile_title, resid=None, max_abs=None):
    if resid is None:
        resid = residual_dataframe(cfg, samples, df)
    if max_abs is None:
        max_abs = residual_axis_limit([resid])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="row")
    axes = axes.flatten()

    for panel_index, ax in enumerate(axes):
        subset = resid[panel_mask(resid, panel_index)].copy()
        if len(subset) == 0:
            ax.set_visible(False)
            continue

        for i, freq in enumerate(np.sort(subset["freq"].unique())):
            rows = subset[np.isclose(subset["freq"], freq)].sort_values("obsdate")
            marker, color = RADIO_BAND_STYLES.get(
                float(freq), (MARKERS[i % len(MARKERS)], COLORS[i % len(COLORS)])
            )
            ax.scatter(
                rows["obsdate"],
                rows["residual_sigma"],
                marker=marker,
                color=color,
                s=32,
                label=freq_label(freq),
            )

        ax.axhline(0, color="0.25", linewidth=1.2)
        ax.axhline(1, color="0.65", linewidth=0.8, linestyle="--")
        ax.axhline(-1, color="0.65", linewidth=0.8, linestyle="--")
        ax.set_title(PANEL_SPECS[panel_index][0], fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(*panel_time_limits(panel_index))
        ax.set_ylim(-max_abs, max_abs)
        ax.set_ylabel(r"Residual ($\sigma$)")
        deduplicate_legend(ax)

    for ax in axes[-2:]:
        ax.set_xlabel("Days post-trigger")

    fig.suptitle(f"{profile_title} Residuals", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=250)
    plt.close(fig)
    print(f"Saved {output}")


def profile_title(profile):
    return "Stellar Wind Profile" if profile == "wind" else "ISM Profile"


def format_profile_path(template, profile):
    if "{profile}" in template:
        return template.format(profile=profile)
    path = Path(template)
    return str(path.with_name(f"{path.stem}_{profile}{path.suffix}"))


def main():
    args = parse_args()
    overrides = mcmc_overrides(args)
    forward_overrides = forward_only_mcmc_overrides(args)
    root_obs = load_root_observation_data()
    check_source = load_check_source_data()
    plot_jobs = []
    statistics_rows = []

    for profile in args.profiles:
        rng = np.random.default_rng(args.seed + (0 if profile == "wind" else 1))
        config_path = resolve_profile_config(args, profile)
        forward_config_path = resolve_forward_only_profile_config(args, profile)
        cfg = load_config(config_path)
        apply_mcmc_overrides(cfg, overrides)
        forward_cfg = load_config(forward_config_path)
        apply_mcmc_overrides(forward_cfg, forward_overrides)
        cache_path = format_profile_path(args.sample_cache, profile)
        output = format_profile_path(args.output, profile)
        residual_output = format_profile_path(args.residual_output, profile)
        _, samples, diagnostics = get_samples(
            cfg,
            config_path,
            cache_path,
            overrides,
            profile,
            force_refit=args.force_refit,
        )
        forward_cache_path = format_profile_path(args.forward_only_sample_cache, profile)
        _, forward_samples, forward_diagnostics = get_samples(
            forward_cfg,
            forward_config_path,
            forward_cache_path,
            forward_overrides,
            f"{profile}:forward_only",
            force_refit=args.force_refit,
        )
        df = radio_data(load_data(cfg))
        print(f"Plotting {profile} profile: {len(df)} data rows with {len(samples)} posterior samples")
        if diagnostics:
            print("Sampler diagnostics:", diagnostics)
        if forward_diagnostics:
            print("Forward-only sampler diagnostics:", forward_diagnostics)

        for model_label, model_cfg, model_samples in [
            ("forward_reverse", cfg, samples),
            ("forward_only", forward_cfg, forward_samples),
        ]:
            metrics = fit_statistics(model_cfg, model_samples, df)
            statistics_rows.append(
                {
                    "profile": profile,
                    "model": model_label,
                    "data": "radio",
                    **metrics,
                }
            )
            print(
                f"{profile} {model_label} radio-only statistics: "
                f"chi-square={metrics['chisq']:.6g}, "
                f"reduced chi-square={metrics['redchisq']:.6g}, "
                f"AIC={metrics['AIC']:.6g}, BIC={metrics['BIC']:.6g}"
            )

        resid = residual_dataframe(cfg, samples, df)
        plot_jobs.append(
            {
                "profile": profile,
                "comparison_df": load_comparison_data(config_path),
                "rng": rng,
                "cfg": cfg,
                "samples": samples,
                "forward_cfg": forward_cfg,
                "forward_samples": forward_samples,
                "df": df,
                "output": output,
                "residual_output": residual_output,
                "resid": resid,
            }
        )

    make_extrapolation_plot(plot_jobs, args.extrapolation_output, args.extrapolation_draws, args.seed)
    write_fit_statistics(statistics_rows, args.fit_statistics_output)
    residual_max_abs = residual_axis_limit([job["resid"] for job in plot_jobs])

    for job in plot_jobs:
        make_lightcurve_plot(
            job["cfg"],
            job["samples"],
            job["df"],
            job["output"],
            args.n_draws,
            job["rng"],
            profile_title(job["profile"]),
            root_obs,
            check_source,
            forward_only_cfg=job["forward_cfg"],
            forward_only_samples=job["forward_samples"],
        )
        make_residual_plot(
            job["cfg"],
            job["samples"],
            job["df"],
            job["residual_output"],
            profile_title(job["profile"]),
            resid=job["resid"],
            max_abs=residual_max_abs,
        )


if __name__ == "__main__":
    main()
