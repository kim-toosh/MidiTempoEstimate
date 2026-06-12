"""Visualisation helpers for the Kalman gate debug workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from midi_tempo_hmm.output.estimator_result import EstimatorResult

# ── colour palette (matches main_gui dark theme) ───────────────────────────
_BG       = '#1E1E2E'
_BG_AX    = '#181825'
_FG       = '#CDD6F4'
_BORDER   = '#45475A'
_RED      = '#E74C3C'
_BLUE     = '#3498DB'
_GREEN    = '#2ECC71'
_YELLOW   = '#F9E2AF'
_ORANGE   = '#FAB387'


def _extract(
    history: list[EstimatorResult],
) -> tuple[list, list, list, list, list, list, list]:
    """Return (indices, raw, gated, predicted, mahal, var, reasons) from history."""
    indices   : list[int]   = []
    raw       : list[float] = []
    gated     : list[float] = []
    predicted : list[float] = []
    mahal     : list[float] = []
    var_      : list[float] = []
    reasons   : list[str]   = []

    for i, r in enumerate(history):
        if r.gate_result is None:
            continue
        g = r.gate_result
        indices.append(i)
        raw.append(g.raw_candidate)
        gated.append(g.gated_tempo)
        predicted.append(g.predicted_tempo)
        mahal.append(g.mahal_distance)
        var_.append(g.current_var)
        reasons.append(g.reject_reason)

    return indices, raw, gated, predicted, mahal, var_, reasons


def plot_gate_debug(
    history: list[EstimatorResult],
    true_tempo: Optional[float] = None,
    block: bool = True,
) -> None:
    """4-panel dark-theme debug plot for the Kalman gate.

    Panels:
      1. Tempo tracks — raw candidate (dim), gated tempo (bright), Kalman
         prediction (dashed), ±2σ band, true tempo line (if given), REJECT×
      2. Mahalanobis distance with gate threshold
      3. Kalman posterior variance
      4. Stacked bar of reject reasons per event

    Args:
        history:    List of :class:`~midi_tempo_hmm.output.estimator_result.EstimatorResult`.
        true_tempo: Known true tempo for RMS error annotation (optional).
        block:      Passed to ``plt.show(block=block)``.
    """
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    indices, raw, gated, predicted, mahal, var_, reasons = _extract(history)
    if not indices:
        print("[gate_visualizer] No gate_result data to plot.")
        return

    xs       = np.array(indices)
    raw_a    = np.array(raw)
    gated_a  = np.array(gated)
    pred_a   = np.array(predicted)
    mahal_a  = np.array(mahal)
    var_a    = np.array(var_)
    sigma2   = np.sqrt(var_a)
    reasons_a = np.array(reasons)

    accepted_mask = reasons_a == ""
    rej_conf_mask = reasons_a == "confidence"
    rej_oct_mask  = reasons_a == "octave"
    rej_mah_mask  = reasons_a == "mahal"

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 10),
        facecolor=_BG,
        gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1]},
    )
    fig.suptitle('Kalman Gate Debug', color=_FG, fontsize=14)

    def _style(ax: plt.Axes) -> None:
        ax.set_facecolor(_BG_AX)
        ax.tick_params(colors=_FG, labelcolor=_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
        ax.grid(True, color=_BORDER, linestyle='--', linewidth=0.4, alpha=0.6)

    # ── Panel 1: tempo tracks ──────────────────────────────────────────────
    ax1 = axes[0]
    _style(ax1)
    ax1.plot(xs, raw_a,   color=_FG,     alpha=0.3, linewidth=1.0, label='raw candidate')
    ax1.plot(xs, gated_a, color=_RED,    alpha=0.9, linewidth=1.5, label='gated tempo')
    ax1.plot(xs, pred_a,  color=_BLUE,   alpha=0.7, linewidth=1.0, linestyle='--', label='Kalman pred')
    ax1.fill_between(xs, pred_a - 2*sigma2, pred_a + 2*sigma2,
                     color=_BLUE, alpha=0.12, label='±2σ')

    if true_tempo is not None:
        ax1.axhline(true_tempo, color=_GREEN, linestyle=':', linewidth=1.2, label=f'true {true_tempo:.1f} BPM')

    # Reject markers
    for mask, colour, label in [
        (rej_conf_mask, _ORANGE, 'reject:confidence'),
        (rej_oct_mask,  _YELLOW, 'reject:octave'),
        (rej_mah_mask,  _RED,    'reject:mahal'),
    ]:
        if mask.any():
            ax1.scatter(xs[mask], raw_a[mask], color=colour, marker='x',
                        s=60, zorder=6, label=label)

    ax1.set_ylabel('BPM', color=_FG)
    ax1.legend(fontsize=8, facecolor=_BG, labelcolor=_FG, framealpha=0.6, loc='upper right')

    # ── Panel 2: Mahalanobis distance ─────────────────────────────────────
    ax2 = axes[1]
    _style(ax2)
    ax2.plot(xs, mahal_a, color=_BLUE, linewidth=1.2)
    if len(mahal) > 0:
        threshold = history[indices[0]].gate_result.mahal_threshold  # type: ignore[union-attr]
        ax2.axhline(threshold, color=_RED, linestyle='--', linewidth=1.0, label=f'threshold ({threshold:.1f})')
        ax2.legend(fontsize=8, facecolor=_BG, labelcolor=_FG, framealpha=0.6)
    ax2.set_ylabel('Mahal²', color=_FG)
    ax2.set_yscale('log')

    # ── Panel 3: Kalman variance ───────────────────────────────────────────
    ax3 = axes[2]
    _style(ax3)
    ax3.plot(xs, var_a, color=_GREEN, linewidth=1.2)
    ax3.set_ylabel('Variance', color=_FG)

    # ── Panel 4: stacked reject-reason bar ────────────────────────────────
    ax4 = axes[3]
    _style(ax4)
    bottom = np.zeros(len(xs))
    for mask, colour, label in [
        (accepted_mask, _GREEN,  'accept'),
        (rej_conf_mask, _ORANGE, 'confidence'),
        (rej_oct_mask,  _YELLOW, 'octave'),
        (rej_mah_mask,  _RED,    'mahal'),
    ]:
        vals = mask.astype(float)
        ax4.bar(xs, vals, bottom=bottom, color=colour, width=1.0, label=label)
        bottom += vals
    ax4.set_ylabel('Reason', color=_FG)
    ax4.set_yticks([])
    ax4.legend(fontsize=7, facecolor=_BG, labelcolor=_FG, framealpha=0.6,
               loc='upper right', ncol=4)

    axes[-1].set_xlabel('Event #', color=_FG)
    fig.tight_layout()
    plt.show(block=block)


def plot_gate_statistics(history: list[EstimatorResult]) -> None:
    """Summary statistics plot: pie chart of reject reasons + Mahal histogram.

    Args:
        history: List of :class:`~midi_tempo_hmm.output.estimator_result.EstimatorResult`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _, _, _, _, mahal, _, reasons = _extract(history)
    if not reasons:
        print("[gate_visualizer] No gate_result data to plot.")
        return

    reasons_a = np.array(reasons)
    n_accept = int((reasons_a == "").sum())
    n_conf   = int((reasons_a == "confidence").sum())
    n_oct    = int((reasons_a == "octave").sum())
    n_mah    = int((reasons_a == "mahal").sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=_BG)
    fig.suptitle('Kalman Gate Statistics', color=_FG, fontsize=13)

    # Pie chart
    ax_pie = axes[0]
    ax_pie.set_facecolor(_BG_AX)
    sizes  = [n_accept, n_conf, n_oct, n_mah]
    labels = ['Accept', 'Reject:confidence', 'Reject:octave', 'Reject:mahal']
    colours = [_GREEN, _ORANGE, _YELLOW, _RED]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colours) if s > 0]
    if non_zero:
        s_nz, l_nz, c_nz = zip(*non_zero)
        ax_pie.pie(s_nz, labels=l_nz, colors=c_nz, autopct='%1.1f%%',
                   textprops={'color': _FG})
    ax_pie.set_title('Reject reasons', color=_FG)

    # Mahal histogram
    ax_hist = axes[1]
    ax_hist.set_facecolor(_BG_AX)
    ax_hist.tick_params(colors=_FG, labelcolor=_FG)
    for spine in ax_hist.spines.values():
        spine.set_edgecolor(_BORDER)
    if mahal:
        mahal_a = np.array(mahal)
        ax_hist.hist(mahal_a, bins=30, color=_BLUE, alpha=0.8, log=True)
        if len(history) > 0:
            for r in history:
                if r.gate_result is not None:
                    ax_hist.axvline(r.gate_result.mahal_threshold, color=_RED,
                                    linestyle='--', label='threshold')
                    break
        ax_hist.legend(fontsize=8, facecolor=_BG, labelcolor=_FG, framealpha=0.6)
    ax_hist.set_xlabel('Mahalanobis²', color=_FG)
    ax_hist.set_ylabel('Count (log)', color=_FG)
    ax_hist.set_title('Mahal distance distribution', color=_FG)

    fig.tight_layout()
    plt.show()


def print_gate_summary(history: list[EstimatorResult]) -> None:
    """Print a text summary of Kalman gate performance to stdout.

    Reports accept/reject counts, reject breakdown, average Mahalanobis
    distance, final Kalman state, and stability comparison (std dev of raw
    candidates vs gated tempos).

    Args:
        history: List of :class:`~midi_tempo_hmm.output.estimator_result.EstimatorResult`.
    """
    import math

    _, raw, gated, _, mahal, var_, reasons = _extract(history)
    n = len(reasons)
    if n == 0:
        print("=== Kalman Gate Summary ===\n  (no gate data)\n")
        return

    n_accept = reasons.count("")
    n_conf   = reasons.count("confidence")
    n_oct    = reasons.count("octave")
    n_mah    = reasons.count("mahal")

    accept_pct = 100.0 * n_accept / n
    avg_mahal  = sum(mahal) / n
    max_mahal  = max(mahal)

    # Stability: std dev of raw vs gated
    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

    raw_std   = _std(raw)
    gated_std = _std(gated)

    last_var  = var_[-1] if var_ else float('nan')
    last_gate = history[-1].gate_result
    last_mean = last_gate.gated_tempo if last_gate else float('nan')

    print("=== Kalman Gate Summary ===")
    print(f"  Total events : {n}")
    print(f"  Accepted     : {n_accept} ({accept_pct:.1f}%)")
    print(f"  Rejected     : {n - n_accept} ({100.0 - accept_pct:.1f}%)")
    if n - n_accept > 0:
        print(f"    confidence : {n_conf}")
        print(f"    octave     : {n_oct}")
        print(f"    mahal      : {n_mah}")
    print()
    print(f"  Avg Mahal²   : {avg_mahal:.2f}")
    print(f"  Max Mahal²   : {max_mahal:.2f}")
    print()
    print(f"  Kalman final mean : {last_mean:.2f} BPM")
    print(f"  Kalman final var  : {last_var:.4f}")
    print()
    print(f"  Tempo std-dev — raw: {raw_std:.2f} BPM  gated: {gated_std:.2f} BPM")
    if raw_std > 0:
        reduction = 100.0 * (1.0 - gated_std / raw_std)
        print(f"  Smoothing reduction: {reduction:.1f}%")
    print()
