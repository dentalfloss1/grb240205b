#!/usr/bin/env python3
"""Plot the saved forward/reverse fits against only the 9 GHz data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import publication_fit_plot as publication
from grbfit.fit import evaluate_model_components


HERE = Path(__file__).resolve().parent
FIT_DIR = HERE.parent / "forwardreverse"
PROFILE = ("wind", "#0072B2", "-")


def load_saved_fit(profile):
    config_path = FIT_DIR / profile / "config.yaml"
    fit_path = FIT_DIR / profile / "model_fit.json"
    cfg = publication.load_config(config_path)
    fit = json.loads(fit_path.read_text())["fit"]
    cfg["fit"]["param_keys"] = fit["param_keys"]
    cfg["fit"]["fixed_params"] = fit["fixed_params"]
    theta = np.asarray([fit["parameters"][key]["value"] for key in fit["param_keys"]])
    return cfg, theta


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    times = np.geomspace(0.01, 200, 500)
    frequencies = np.full_like(times, 9.0)

    profile, color, linestyle = PROFILE
    cfg, theta = load_saved_fit(profile)
    data = publication.radio_data(publication.load_data(cfg))
    data = data[np.isclose(data["freq"], 9.0)].copy()
    publication.plot_data(ax, data)

    model = evaluate_model_components(cfg, theta, (times, frequencies))["total"] * 1e6
    valid = np.isfinite(model) & (model > 0)
    ax.plot(times[valid], model[valid], color=color, linestyle=linestyle,
            linewidth=2.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 200)
    ax.set_xlabel("Days post-trigger")
    ax.set_ylabel(r"Flux Density ($\mu$Jy)")
    ax.set_title("GRB 240205B (9 GHz)")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    output = HERE / "publication_9ghz_forward_reverse.png"
    fig.savefig(output, dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
