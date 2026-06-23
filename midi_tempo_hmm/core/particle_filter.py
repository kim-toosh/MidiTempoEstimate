"""Particle filter for real-time MIDI tempo, beat, and measure estimation."""

from __future__ import annotations

import dataclasses
import logging
import time
from types import ModuleType
from typing import Optional

import numpy as np

from midi_tempo_hmm.core.autocorr_estimator import AutocorrEstimator
from midi_tempo_hmm.core.instrument_category import CategorizedEvent, InstrumentCategory
from midi_tempo_hmm.core.kalman_gate import KalmanGate
from midi_tempo_hmm.core.midi_input_gate import MidiInputGate
from midi_tempo_hmm.core.observation import compute_likelihood
from midi_tempo_hmm.core.resampling import compute_ess, systematic_resample
from midi_tempo_hmm.core.state import make_particle_arrays
from midi_tempo_hmm.output.estimator_result import EstimatorResult

logger = logging.getLogger(__name__)


class ParticleFilter:
    """Sequential Monte Carlo estimator for MIDI tempo, beat phase, and meter.

    Each particle represents a hypothesis about the current tempo (BPM),
    position within the beat (beat_phase ∈ [0, 1)), and position within the
    measure (meter_phase ∈ {0, …, METER_NUMERATOR-1}).

    Call :meth:`update` once per received MIDI note-on event.  The first call
    for each instrument category returns *None*; subsequent calls of the same
    category perform the full predict–weight–resample cycle.

    Per-category observation model
    --------------------------------
    Events are classified into :class:`InstrumentCategory` (KICK, SNARE, HIHAT,
    OTHERS) by the internal :class:`MidiInputGate`.  Each category has its own
    beat-ratio set optimised for its rhythmic role, stored in *config* as
    ``{CATEGORY}_BEAT_RATIOS`` / ``{CATEGORY}_BEAT_WEIGHTS``.

    Same-category IOI tracking
    ---------------------------
    IOI is measured between consecutive events of the *same* category
    (Kick→Kick, Snare→Snare, …).  Cross-category intervals are intentionally
    ignored to prevent different rhythmic roles from interfering with each other.

    Backward compatibility
    ----------------------
    Calling ``update(ts)`` without *note_number* / *channel* routes the event
    through the OTHERS model with drum_weight = 1.0, giving the same effective
    behaviour as the original non-drum design.

    Args:
        config: Module containing global constants.
        weight_map: Reserved; currently used only by the autocorr estimator.
    """

    _EXPLORER_FRACTION: float = 0.02
    _SEED_FRACTION:     float = 0.15
    _SEED_SPREAD_BPM:   float = 5.0

    def __init__(
        self,
        config: ModuleType,
        weight_map: Optional[dict[int, float]] = None,
    ) -> None:
        self.config = config
        self.n_particles: int = config.N_PARTICLES
        self._custom_weight_map: Optional[dict[int, float]] = weight_map

        # Minimum IOI: half the smallest expected interval at TEMPO_MAX
        all_ratios: list[float] = list(getattr(config, 'BEAT_RATIOS', [0.25]))
        for attr in ('HIHAT_BEAT_RATIOS', 'KICK_BEAT_RATIOS',
                     'SNARE_BEAT_RATIOS', 'OTHERS_BEAT_RATIOS'):
            vals = getattr(config, attr, None)
            if vals:
                all_ratios.extend(vals)
        self._min_ioi_sec: float = (60.0 / config.TEMPO_MAX) * min(all_ratios) * 0.5

        self._converged: bool = False
        self._gcd_seeded: bool = False
        self._prev_tempo: Optional[float] = None
        self._kalman_reject_streak: int = 0

        # 同時イベントのグルーピング: span_same_time以内の異なるノートのイベントを
        # 1つのグループとみなし、確定時にタイムスタンプの平均値を使う。
        self.span_same_time: float = getattr(config, 'SPAN_SAME_TIME_SEC', 0.05)
        self._pending_group: list[tuple[float, int, int, int]] = []
        self._pending_group_start: Optional[float] = None

        use_autocorr = getattr(config, 'USE_DRUM_WEIGHTS', True)
        self._autocorr: Optional[AutocorrEstimator] = (
            AutocorrEstimator(config, weight_map=weight_map) if use_autocorr else None
        )

        self.midi_gate   = MidiInputGate(config)
        self.kalman_gate = KalmanGate(config)

        self._initialize_particles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp_sec: float,
        note_number:   int = 60,
        velocity:      int = 100,
        channel:       int = 0,
    ) -> Optional[EstimatorResult]:
        """Process a MIDI note-on event and return the current tempo estimate.

        Events are classified by :class:`MidiInputGate` into an instrument
        category.  The first event of each category stores the reference
        timestamp and returns *None*.  Subsequent same-category events trigger
        a full predict–weight–resample cycle.

        velocity == 0 (NOTE_OFF) is silently skipped and returns *None*.

        Same-timing event grouping
        ---------------------------
        Events that arrive within ``span_same_time`` seconds of the current
        pending group's first event, and whose note number is not already in
        that group, are buffered rather than processed immediately. When the
        group is finalised (either because a later event falls outside the
        window / repeats a note already in the group, or via :meth:`flush`),
        every buffered event is processed using the AVERAGE timestamp of the
        group. This call returns the result of finalising the *previous*
        group, if any; the just-received event is always buffered into the
        (possibly new) pending group and its own result is deferred.

        Args:
            timestamp_sec: Absolute timestamp in seconds.
            note_number:   MIDI note number (0–127).  Defaults to 60 (middle C),
                           which maps to the OTHERS category on non-drum channels.
            velocity:      MIDI velocity (1–127).  0 is treated as NOTE_OFF.
            channel:       MIDI channel (0-indexed).  Channel 9 is the GM drum channel.

        Returns:
            :class:`EstimatorResult`, or *None* when no group was finalised.
        """
        if velocity == 0:
            return None   # NOTE_OFF

        # 生のタイムスタンプでautocorrバッファに即時フィード（グルーピングと無関係）
        if self._autocorr is not None:
            self._autocorr.add_event(timestamp_sec, note_number, velocity)

        # ── 同時イベントのグルーピング ────────────────────────────────────────
        finalized_result: Optional[EstimatorResult] = None
        if self._pending_group:
            elapsed = timestamp_sec - self._pending_group_start
            pending_notes = {n for (_, n, _, _) in self._pending_group}
            if elapsed > self.span_same_time or note_number in pending_notes:
                finalized_result = self._finalize_pending_group()

        if not self._pending_group:
            self._pending_group_start = timestamp_sec
        self._pending_group.append((timestamp_sec, note_number, velocity, channel))

        return finalized_result

    def flush(self, now_sec: float) -> Optional[EstimatorResult]:
        """Finalise the pending group if it has aged past ``span_same_time``.

        This handles isolated events that never receive a follow-up event
        within ``span_same_time`` — without a periodic call to this method
        such a group would never be finalised.

        Args:
            now_sec: Current time in the same clock domain as the timestamps
                     passed to :meth:`update`.

        Returns:
            :class:`EstimatorResult`, or *None* if there is nothing to flush
            or the pending group has not yet aged past ``span_same_time``.
        """
        if (self._pending_group
                and (now_sec - self._pending_group_start) > self.span_same_time):
            return self._finalize_pending_group()
        return None

    def _finalize_pending_group(self) -> Optional[EstimatorResult]:
        """保留グループを確定し、平均タイムスタンプで各イベントを処理する。"""
        events = self._pending_group
        self._pending_group = []
        self._pending_group_start = None

        avg_ts = float(np.mean([ts for ts, _, _, _ in events]))

        last_result: Optional[EstimatorResult] = None
        for _, note_number, velocity, channel in events:
            result = self._process_single_event(avg_ts, note_number, velocity, channel)
            if result is not None:
                last_result = result
        return last_result

    def _process_single_event(
        self,
        timestamp_sec: float,
        note_number:   int,
        velocity:      int,
        channel:       int,
    ) -> Optional[EstimatorResult]:
        """単一イベント（確定済みグループの平均タイムスタンプを含む）を処理する。

        Runs the full predict–weight–resample–estimate cycle for one event,
        plus GCD reinit, convergence tracking, Kalman gating, and autocorr
        seeding. Returns *None* when the event's same-category IOI is
        missing, non-positive, or below the minimum threshold.
        """
        t0 = time.perf_counter()

        # ── 1. Classify event ─────────────────────────────────────────────────
        event, ioi_sec, gcd_candidates = self.midi_gate.process(
            timestamp_sec, note_number, velocity, channel
        )
        if gcd_candidates:
            # Use KALMAN_INIT_TEMPO as reference so autocorr drift before GCD fires
            # doesn't cause the wrong octave to be selected for the reinit.
            init_ref  = getattr(self.config, 'KALMAN_INIT_TEMPO', 120.0)
            dists     = [abs(t - init_ref) for t, _ in gcd_candidates]
            best_i    = int(dists.index(min(dists)))
            gcd_tempo = gcd_candidates[best_i][0]
            gcd_conf  = gcd_candidates[best_i][1]
        else:
            gcd_tempo = None
            gcd_conf  = 0.0
        if event is None:
            return None   # velocity == 0 (NOTE_OFF) — should not occur here

        # ── 2. Same-category IOI ──────────────────────────────────────────────
        logger.debug("ioi(%s) -> %s", event.category.name, ioi_sec)
        if ioi_sec is None:
            logger.debug(
                "First %s event at %.4f s — reference time set.",
                event.category.name, timestamp_sec,
            )
            return None

        if ioi_sec <= 0.0:
            logger.warning("Non-positive IOI (%.6f s) ignored.", ioi_sec)
            return None

        if ioi_sec < self._min_ioi_sec:
            logger.debug(
                "IOI %.6f s below minimum %.6f s; skipped.",
                ioi_sec, self._min_ioi_sec,
            )
            return None

        # ── 3. Particle filter cycle ──────────────────────────────────────────
        self._predict(ioi_sec)
        self._update_weights(ioi_sec, event)
        pre_resample_ess = self._resample_if_needed()

        processing_time_ms = (time.perf_counter() - t0) * 1000.0
        result = self._estimate(processing_time_ms, pre_resample_ess)

        # ── 4. GCDによる粒子再初期化（初回のみ）───────────────────────────────
        # 収束判定（4b）より先に行う。GCDが最初に信頼度を満たすイベントが
        # HIHATになることがあり、4bのHIHAT例外（gcd_reliable）が先に収束扱いに
        # してしまうと、この再初期化が永久にブロックされてしまうため。
        gcd_reliable = gcd_conf >= getattr(self.config, 'GCD_CONFIDENCE_THRESHOLD', 0.70)
        if (not self._converged
                and not self._gcd_seeded
                and gcd_tempo is not None
                and gcd_reliable):
            # gcd_tempoは_correct_gcd_octave()で既にオクターブ補正済み。
            self._reinit_from_tempo(gcd_tempo, sigma=getattr(self.config, 'GCD_REINIT_SIGMA', 3.0))
            self._gcd_seeded = True
            logger.debug(
                "GCD reinit: %.1f BPM (confidence=%.2f)",
                gcd_tempo, gcd_conf
            )

        # ── 4b. Convergence check ────────────────────────────────────────────
        # Normally block HIHAT events: skipped weight update leaves uniform weights
        # (ESS=N → confidence=1.0), which would falsely declare convergence.
        # Exception: allow HIHAT when GCD has high confidence — in that case
        # the uniform weights reflect a genuine post-reinit high-confidence state.
        if (not self._converged
                and result.confidence >= self.config.CONVERGENCE_THRESHOLD
                and (event.category != InstrumentCategory.HIHAT or gcd_reliable)):
            self._converged = True
            logger.debug("Particle filter converged at confidence=%.3f.", result.confidence)
        elif self._converged and result.confidence < self.config.CONVERGENCE_THRESHOLD * 0.5:
            self._converged = False
            logger.debug(
                "De-converged: confidence=%.3f dropped below de-convergence threshold.",
                result.confidence,
            )

        # ── 5. Kalman gate (+ streak-based feedback seeding) ─────────────────
        gate_result = self.kalman_gate.update([(result.tempo_bpm, result.confidence)])

        if not self._converged:
            if gate_result.accepted:
                self._kalman_reject_streak = 0
            else:
                self._kalman_reject_streak += 1
                streak_limit = getattr(self.config, 'KALMAN_SEED_STREAK', 5)
                if self._kalman_reject_streak >= streak_limit:
                    self._seed_from_kalman()

        # ── 6. Autocorr seeding (pre-convergence, non-HIHAT events only) ─────
        # Restrict seeding to kick/snare/others: HIHAT's dense subdivisions produce
        # IOIs that dominate the autocorr buffer and map to the wrong octave
        # (e.g. 16th notes at 80 BPM → 320 BPM → seeds at 160 BPM, not 80).
        autocorr_tempo: Optional[float] = None
        if not self._converged and self._autocorr is not None and event.category != InstrumentCategory.HIHAT:
            candidates = self._autocorr.estimate()
            if candidates:
                best_bpm, strength = candidates[0]
                # Prefer the octave-corrected version closest to the current estimate
                seed_bpm = best_bpm
                for ratio in (0.25, 0.5, 2.0, 4.0):
                    alt = best_bpm * ratio
                    if (self.config.TEMPO_MIN <= alt <= self.config.TEMPO_MAX
                            and abs(alt - result.tempo_bpm) < abs(seed_bpm - result.tempo_bpm)):
                        seed_bpm = alt
                autocorr_tempo = seed_bpm
                self._seed_from_autocorr(seed_bpm)
                logger.debug(
                    "Autocorr seed: %.1f BPM (raw=%.1f, strength=%.2f).",
                    seed_bpm, best_bpm, strength,
                )

        # ── 7. Assemble result ────────────────────────────────────────────────
        result = dataclasses.replace(
            result,
            is_converged=self._converged,
            autocorr_tempo=autocorr_tempo,
            gate_result=gate_result,
            last_category=event.category,
            category_counts=self.midi_gate.get_stats(),
            gcd_tempo=gcd_tempo,
            gcd_confidence=gcd_conf,
            ioi_sec=ioi_sec,
            note_number=note_number,
            gcd_iois=list(self.midi_gate.last_gcd_iois),
        )

        # Jump detection
        if self._prev_tempo is not None:
            delta = result.tempo_bpm - self._prev_tempo
            if abs(delta) >= 30.0:
                logger.debug(
                    "Large tempo jump: %.2f → %.2f BPM (delta=%+.2f)  "
                    "ioi=%.4f s  conf=%.3f  converged=%s  cat=%s",
                    self._prev_tempo, result.tempo_bpm, delta,
                    ioi_sec, result.confidence, self._converged,
                    event.category.name,
                )
        self._prev_tempo = result.tempo_bpm

        logger.debug(
            "tempo=%.2f BPM  beat_pos=%.3f  measure_beat=%d  conf=%.3f  "
            "cat=%s  t=%.2f ms",
            result.tempo_bpm, result.beat_position,
            result.measure_beat, result.confidence,
            event.category.name, result.processing_time_ms,
        )
        return result

    def reset(self) -> None:
        """Re-initialise all particles and clear all state."""
        self._converged = False
        self._gcd_seeded = False
        self._prev_tempo = None
        self._kalman_reject_streak = 0
        self._pending_group = []
        self._pending_group_start = None
        if self._autocorr is not None:
            self._autocorr.reset()
        self.midi_gate.reset()
        self.kalman_gate.reset()
        self._initialize_particles()

    def set_weight_map(self, weight_map: dict[int, float]) -> None:
        """Replace the custom drum weight map on the autocorr estimator."""
        self._custom_weight_map = weight_map
        if self._autocorr is not None:
            self._autocorr.set_weight_map(weight_map)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _initialize_particles(self) -> None:
        (
            self.tempos,
            self.beat_phases,
            self.meter_phases,
            self.weights,
        ) = make_particle_arrays(
            self.n_particles,
            self.config.TEMPO_MIN,
            self.config.TEMPO_MAX,
        )

    def _predict(self, ioi_sec: float) -> None:
        noise = np.random.normal(0.0, self.config.SIGMA_TEMPO, self.n_particles)
        self.tempos += noise
        np.clip(self.tempos, self.config.TEMPO_MIN, self.config.TEMPO_MAX, out=self.tempos)

        n_explorers = max(1, int(self._EXPLORER_FRACTION * self.n_particles))
        explorer_idx = np.random.choice(self.n_particles, n_explorers, replace=False)
        self.tempos[explorer_idx] = np.random.uniform(
            self.config.TEMPO_MIN, self.config.TEMPO_MAX, n_explorers
        )
        self.beat_phases[explorer_idx] = np.random.uniform(0.0, 1.0, n_explorers)

        beat_periods  = 60.0 / self.tempos
        phase_advance = ioi_sec / beat_periods
        self.beat_phases += phase_advance

        beat_crossings = np.floor(self.beat_phases).astype(np.int32)
        self.meter_phases = (self.meter_phases + beat_crossings) % self.config.METER_NUMERATOR
        self.beat_phases  = self.beat_phases % 1.0

    def _update_weights(
        self,
        ioi_sec: float,
        event: CategorizedEvent,
    ) -> None:
        if event.category == InstrumentCategory.HIHAT:
            if not self._converged:
                return
            # Post-convergence HIHAT: skip short subdivisions (2x-ambiguous)
            estimated_bpm = float(np.dot(self.weights, self.tempos))
            beat_period = 60.0 / estimated_bpm
            if ioi_sec < 0.45 * beat_period:
                return

        likelihoods = compute_likelihood(
            ioi_sec, self.tempos, event.category, self.config, event.drum_weight
        )

        self.weights *= likelihoods
        total = self.weights.sum()
        if total > 0.0:
            self.weights /= total
        else:
            self.weights[:] = 1.0 / self.n_particles
            logger.warning("Weight collapse detected; resetting to uniform.")

    def _resample_if_needed(self) -> float:
        """Resample if ESS is below threshold. Returns the pre-resample ESS.

        The pre-resample ESS is used for confidence estimation so that a weight
        collapse (low ESS → resample → uniform weights → ESS=N) does not
        incorrectly report high confidence.
        """
        ess = compute_ess(self.weights)
        threshold = self.n_particles * self.config.ESS_THRESHOLD_RATIO
        if ess < threshold:
            indices = systematic_resample(self.weights)
            self.tempos       = self.tempos[indices]
            self.beat_phases  = self.beat_phases[indices]
            self.meter_phases = self.meter_phases[indices]
            self.weights      = np.ones(self.n_particles) / self.n_particles
            logger.debug("Resampled (ESS=%.1f < threshold=%.1f).", ess, threshold)
        return ess

    def _seed_from_kalman(self) -> None:
        target_bpm = self.kalman_gate.mean
        n_seed     = max(1, int(self._SEED_FRACTION * self.n_particles))
        seed_idx   = np.argsort(self.weights)[:n_seed]
        spread     = getattr(self.config, 'KALMAN_SEED_SPREAD_BPM', 3.0)

        new_tempos = np.random.normal(target_bpm, spread, n_seed)
        np.clip(new_tempos, self.config.TEMPO_MIN, self.config.TEMPO_MAX, out=new_tempos)
        self.tempos[seed_idx]      = new_tempos
        self.beat_phases[seed_idx] = np.random.uniform(0.0, 1.0, n_seed)
        logger.debug(
            "Kalman feedback seed: %.1f BPM (reject_streak=%d)",
            target_bpm, self._kalman_reject_streak,
        )

    def _seed_from_autocorr(self, target_bpm: float) -> None:
        n_seed   = max(1, int(self._SEED_FRACTION * self.n_particles))
        seed_idx = np.argsort(self.weights)[:n_seed]
        new_tempos = np.random.normal(target_bpm, self._SEED_SPREAD_BPM, n_seed)
        np.clip(new_tempos, self.config.TEMPO_MIN, self.config.TEMPO_MAX, out=new_tempos)
        self.tempos[seed_idx]      = new_tempos
        self.beat_phases[seed_idx] = np.random.uniform(0.0, 1.0, n_seed)

    def _correct_gcd_octave(self, gcd_tempo: float) -> float:
        """gcd_tempoをGCD_OCTAVE_RATIOSで割り、Kalman平均に最も近い候補を返す。"""
        octave_ratios    = np.asarray(self.config.GCD_OCTAVE_RATIOS, dtype=np.float64)
        candidate_tempos = gcd_tempo / octave_ratios
        return float(candidate_tempos[np.argmin(np.abs(candidate_tempos - self.kalman_gate.mean))])

    def _reinit_from_tempo(self, tempo: float, sigma: float) -> None:
        """指定テンポの周辺に全粒子を再配置する。"""
        new_tempos = np.random.normal(tempo, sigma, self.n_particles)
        np.clip(new_tempos, self.config.TEMPO_MIN, self.config.TEMPO_MAX, out=new_tempos)
        self.tempos      = new_tempos
        self.beat_phases = np.random.uniform(0.0, 1.0, self.n_particles)
        self.weights     = np.ones(self.n_particles) / self.n_particles

    def _estimate(self, processing_time_ms: float, pre_resample_ess: float) -> EstimatorResult:
        tempo = float(np.dot(self.weights, self.tempos))
        tempo = round(tempo, self.config.TEMPO_PRECISION)

        angles       = 2.0 * np.pi * self.beat_phases
        sin_mean     = float(np.dot(self.weights, np.sin(angles)))
        cos_mean     = float(np.dot(self.weights, np.cos(angles)))
        beat_position = float(np.arctan2(sin_mean, cos_mean) / (2.0 * np.pi)) % 1.0

        counts = np.bincount(
            self.meter_phases,
            weights=self.weights,
            minlength=self.config.METER_NUMERATOR,
        )
        measure_beat = int(np.argmax(counts))

        confidence = float(pre_resample_ess / self.n_particles)

        return EstimatorResult(
            tempo_bpm=tempo,
            beat_position=beat_position,
            measure_beat=measure_beat,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )
