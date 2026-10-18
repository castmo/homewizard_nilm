"""Machine Learning & NILM Event Disaggregation Engine for HomeWizard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import math
import re
from typing import Any
import uuid

from .const import (
    CONSUMER_UNKNOWN_ID,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONSUMERS,
    DEFAULT_MIN_POWER_DELTA,
    DEFAULT_STABILIZATION_TIME,
)

_LOGGER = logging.getLogger(__name__)


def format_event_time(timestamp: float) -> str:
    """Format Unix timestamp into clean HH:MM:SS UTC time string."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%H:%M:%S UTC")


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a compact human-readable string."""
    secs = max(0, int(round(seconds)))
    if secs < 60:
        return f"{secs}s"
    minutes = secs // 60
    rem_secs = secs % 60
    if minutes < 60:
        if rem_secs > 0:
            return f"{minutes}m {rem_secs}s"
        return f"{minutes}m"
    hours = minutes // 60
    rem_mins = minutes % 60
    return f"{hours}h {rem_mins}m"


def format_energy(kwh: float) -> str:
    """Format energy consumption in both kWh and Wh."""
    wh = kwh * 1000.0
    if kwh < 0.01:
        return f"{wh:.1f} Wh ({kwh:.4f} kWh)"
    if kwh < 1.0:
        return f"{kwh:.3f} kWh ({wh:.0f} Wh)"
    return f"{kwh:.2f} kWh"


def slugify_consumer_name(name: str) -> str:
    """Convert a human-readable consumer name into a clean snake_case ID."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"['’`]", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_") or f"consumer_{uuid.uuid4().hex[:6]}"


@dataclass
class PowerSample:
    """Represents a single power reading from HomeWizard sensors."""

    timestamp: float  # Unix timestamp in seconds
    total_w: float
    l1_w: float | None = None
    l2_w: float | None = None
    l3_w: float | None = None

    @property
    def has_three_phase(self) -> bool:
        """Return True if all 3 phase readings are present."""
        return (
            self.l1_w is not None
            and self.l2_w is not None
            and self.l3_w is not None
        )


@dataclass
class SignatureFeatures:
    """Extracted electrical signature features of an appliance turn-ON event."""

    delta_p: float  # Steady-state step change in Watts
    peak_overshoot: float  # Transient inrush peak above steady-state step (W)
    steady_std: float  # Standard deviation of power after turn-on (W)
    l1_share: float | None = None  # Fraction of delta_p on L1 [0.0 .. 1.0]
    l2_share: float | None = None  # Fraction of delta_p on L2 [0.0 .. 1.0]
    l3_share: float | None = None  # Fraction of delta_p on L3 [0.0 .. 1.0]
    hour_sin: float = 0.0  # Cyclical time-of-day sine
    hour_cos: float = 1.0  # Cyclical time-of-day cosine

    def to_dict(self) -> dict[str, Any]:
        """Serialize features to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignatureFeatures:
        """Deserialize features from dict."""
        return cls(
            delta_p=float(data.get("delta_p", 0.0)),
            peak_overshoot=float(data.get("peak_overshoot", 0.0)),
            steady_std=float(data.get("steady_std", 0.0)),
            l1_share=float(data["l1_share"]) if data.get("l1_share") is not None else None,
            l2_share=float(data["l2_share"]) if data.get("l2_share") is not None else None,
            l3_share=float(data["l3_share"]) if data.get("l3_share") is not None else None,
            hour_sin=float(data.get("hour_sin", 0.0)),
            hour_cos=float(data.get("hour_cos", 1.0)),
        )

    @property
    def dominant_phase(self) -> str:
        """Return a human-readable description of the active phase(s)."""
        if self.l1_share is None or self.l2_share is None or self.l3_share is None:
            return "Single/Total"
        shares = [("L1", self.l1_share), ("L2", self.l2_share), ("L3", self.l3_share)]
        active = [name for name, val in shares if val >= 0.20]
        if not active:
            best = max(shares, key=lambda item: item[1])[0]
            return best
        return "+".join(active)


@dataclass
class TrainingSample:
    """A user-verified or high-confidence labeled appliance signature."""

    consumer_id: str
    features: SignatureFeatures
    created_at: str


@dataclass
class PredictionResult:
    """Output of the ML appliance classifier."""

    predicted_consumer_id: str | None
    confidence: float
    candidates: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ApplianceEvent:
    """Represents a detected appliance turn-ON session (active or completed)."""

    event_id: str
    started_at: float
    started_iso: str
    features: SignatureFeatures
    predicted_consumer_id: str | None
    confidence: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    assigned_consumer_id: str | None = None  # None while pending user input
    is_active: bool = True
    ended_at: float | None = None
    accumulated_kwh: float = 0.0
    last_integrated_ts: float = 0.0

    def duration_seconds(self, now: float | None = None) -> float:
        """Calculate duration in seconds since appliance turned on."""
        if not self.is_active and self.ended_at is not None:
            end = self.ended_at
        elif now is not None:
            end = now
        elif self.last_integrated_ts > 0.0:
            end = self.last_integrated_ts
        else:
            end = datetime.now(timezone.utc).timestamp()
        return max(0.0, end - self.started_at)

    def duration_str(self, now: float | None = None) -> str:
        """Return human-readable duration since appliance turned on."""
        return format_duration(self.duration_seconds(now))

    def started_time_str(self) -> str:
        """Return clean start time in HH:MM:SS UTC."""
        return format_event_time(self.started_at)

    def consumed_energy_str(self) -> str:
        """Return human-readable consumed energy string."""
        return format_energy(self.accumulated_kwh)

    def to_summary_dict(self, consumers: dict[str, str]) -> dict[str, Any]:
        """Return a user-friendly dictionary representation for HA attributes."""
        pred_name = (
            consumers.get(self.predicted_consumer_id, self.predicted_consumer_id)
            if self.predicted_consumer_id
            else "None (Cold Start)"
        )
        assigned_name = (
            consumers.get(self.assigned_consumer_id, self.assigned_consumer_id)
            if self.assigned_consumer_id
            else "Pending User Confirmation"
        )
        return {
            "event_id": self.event_id,
            "started_at": self.started_iso,
            "started_time": self.started_time_str(),
            "duration_seconds": round(self.duration_seconds()),
            "duration": self.duration_str(),
            "power_delta_w": round(self.features.delta_p, 1),
            "peak_overshoot_w": round(self.features.peak_overshoot, 1),
            "steady_std_w": round(self.features.steady_std, 1),
            "phase": self.features.dominant_phase,
            "predicted_consumer_id": self.predicted_consumer_id,
            "predicted_consumer": pred_name,
            "confidence": round(self.confidence, 3),
            "confidence_pct": round(self.confidence * 100, 1),
            "assigned_consumer_id": self.assigned_consumer_id,
            "assigned_consumer": assigned_name,
            "is_active": self.is_active,
            "accumulated_wh": round(self.accumulated_kwh * 1000.0, 1),
            "accumulated_kwh": round(self.accumulated_kwh, 4),
            "consumed_energy": self.consumed_energy_str(),
            "top_candidates": [
                {
                    "consumer_id": cid,
                    "name": consumers.get(cid, cid),
                    "score": round(score, 3),
                }
                for cid, score in self.candidates[:4]
            ],
        }


class ApplianceClassifier:
    """Online Weighted k-NN + RBF Kernel Prototype Classifier for NILM signatures."""

    def __init__(self, max_samples_per_consumer: int = 100) -> None:
        """Initialize the classifier."""
        self.samples: list[TrainingSample] = []
        self.max_samples_per_consumer = max_samples_per_consumer

    def load_samples(self, raw_samples: list[dict[str, Any]]) -> None:
        """Load serialized training samples from storage."""
        loaded: list[TrainingSample] = []
        for item in raw_samples:
            try:
                loaded.append(
                    TrainingSample(
                        consumer_id=str(item["consumer_id"]),
                        features=SignatureFeatures.from_dict(item["features"]),
                        created_at=str(item.get("created_at", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self.samples = loaded

    def dump_samples(self) -> list[dict[str, Any]]:
        """Serialize training samples for storage."""
        return [
            {
                "consumer_id": s.consumer_id,
                "features": s.features.to_dict(),
                "created_at": s.created_at,
            }
            for s in self.samples
        ]

    def add_training_sample(
        self, consumer_id: str, features: SignatureFeatures, created_at: str | None = None
    ) -> None:
        """Add a labeled signature sample and cap per-class history."""
        if consumer_id == CONSUMER_UNKNOWN_ID:
            return
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        self.samples.append(
            TrainingSample(
                consumer_id=consumer_id,
                features=features,
                created_at=created_at,
            )
        )
        # Keep at most max_samples_per_consumer for this consumer (FIFO)
        consumer_indices = [
            idx for idx, s in enumerate(self.samples) if s.consumer_id == consumer_id
        ]
        if len(consumer_indices) > self.max_samples_per_consumer:
            drop_count = len(consumer_indices) - self.max_samples_per_consumer
            for idx in reversed(consumer_indices[:drop_count]):
                self.samples.pop(idx)

    @staticmethod
    def compute_distance(f1: SignatureFeatures, f2: SignatureFeatures) -> float:
        """Compute normalized multi-dimensional signature distance.

        A distance of 0.0 means identical signature; ~1.0 means significantly different.
        """
        # 1. Relative and absolute active power delta distance (most important feature)
        max_p = max(abs(f1.delta_p), abs(f2.delta_p), 100.0)
        rel_p_diff = abs(f1.delta_p - f2.delta_p) / max_p
        # Also penalize absolute difference so 150W vs 300W is clearly distinct
        abs_p_diff = abs(f1.delta_p - f2.delta_p) / 400.0
        p_dist = 0.65 * rel_p_diff + 0.35 * min(abs_p_diff, 2.0)

        # 2. Transient peak overshoot distance (inrush current signature)
        norm_overshoot_1 = f1.peak_overshoot / max(abs(f1.delta_p), 100.0)
        norm_overshoot_2 = f2.peak_overshoot / max(abs(f2.delta_p), 100.0)
        overshoot_dist = abs(norm_overshoot_1 - norm_overshoot_2)

        # 3. Steady-state ripple/variance distance (resistive vs motor/pulsing load)
        norm_std_1 = f1.steady_std / max(abs(f1.delta_p), 100.0)
        norm_std_2 = f2.steady_std / max(abs(f2.delta_p), 100.0)
        std_dist = abs(norm_std_1 - norm_std_2)

        # 4. Cyclical time-of-day distance [0.0 .. 1.0]
        tod_euclid = math.hypot(f1.hour_sin - f2.hour_sin, f1.hour_cos - f2.hour_cos)
        tod_dist = tod_euclid / 2.0

        # 5. Three-phase distribution distance (if both samples have 3-phase data)
        if (
            f1.l1_share is not None
            and f1.l2_share is not None
            and f1.l3_share is not None
            and f2.l1_share is not None
            and f2.l2_share is not None
            and f2.l3_share is not None
        ):
            phase_dist = (
                abs(f1.l1_share - f2.l1_share)
                + abs(f1.l2_share - f2.l2_share)
                + abs(f1.l3_share - f2.l3_share)
            ) / 2.0
            # When 3-phase is available, phase + power delta carry the primary weight
            total_dist = (
                2.2 * p_dist
                + 1.6 * phase_dist
                + 0.35 * overshoot_dist
                + 0.35 * std_dist
                + 0.08 * tod_dist
            )
        else:
            total_dist = (
                2.6 * p_dist
                + 0.45 * overshoot_dist
                + 0.45 * std_dist
                + 0.10 * tod_dist
            )

        return total_dist

    def predict(self, features: SignatureFeatures, k: int = 5) -> PredictionResult:
        """Predict which consumer matches the extracted signature features.

        Returns the predicted consumer ID, a calibrated confidence in [0.0, 1.0],
        and a sorted list of candidate (consumer_id, score) tuples.
        """
        if not self.samples:
            return PredictionResult(predicted_consumer_id=None, confidence=0.0, candidates=[])

        # Compute similarity for every training sample using an RBF kernel
        sigma = 0.45
        scored_samples: list[tuple[str, float, float]] = []
        for sample in self.samples:
            dist = self.compute_distance(features, sample.features)
            sim = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
            scored_samples.append((sample.consumer_id, sim, dist))

        # Sort by highest similarity first
        scored_samples.sort(key=lambda item: item[1], reverse=True)

        # Group top-k similarities per class
        class_sims: dict[str, list[float]] = {}
        for cid, sim, _ in scored_samples:
            bucket = class_sims.setdefault(cid, [])
            if len(bucket) < k:
                bucket.append(sim)

        # Compute class score: combination of best similarity and mean of top-k neighbors
        # plus a maturity factor so a single training sample has bounded confidence (~0.72)
        # while >=2 consistent samples easily reach 0.80 - 0.98 confidence.
        class_scores: dict[str, float] = {}
        class_best_sim: dict[str, float] = {}
        for cid, sims in class_sims.items():
            best_sim = sims[0]
            mean_sim = sum(sims) / len(sims)
            n_samples = len(sims)
            # Maturity factor: 1 sample -> 0.84, 2 samples -> 0.95, 3+ samples -> 1.0
            maturity = min(1.0, 0.73 + 0.11 * n_samples)
            class_best_sim[cid] = best_sim
            class_scores[cid] = (0.65 * best_sim + 0.35 * mean_sim) * maturity

        sorted_candidates = sorted(
            class_scores.items(), key=lambda item: item[1], reverse=True
        )
        best_cid, best_score = sorted_candidates[0]
        best_raw_sim = class_best_sim[best_cid]

        # If even the closest training sample is very far away, return low confidence
        if best_raw_sim < 0.25:
            return PredictionResult(
                predicted_consumer_id=best_cid,
                confidence=round(best_score * 0.5, 4),
                candidates=sorted_candidates,
            )

        # Ambiguity penalty if a second consumer has nearly identical signature score
        if len(sorted_candidates) > 1:
            second_score = sorted_candidates[1][1]
            # Margin ratio between best and runner-up
            margin = max(0.0, best_score - second_score) / max(best_score, 1e-6)
            ambiguity_factor = 0.65 + 0.35 * min(1.0, margin / 0.25)
            confidence = best_score * ambiguity_factor
        else:
            confidence = best_score

        confidence = max(0.0, min(1.0, confidence))
        return PredictionResult(
            predicted_consumer_id=best_cid,
            confidence=round(confidence, 4),
            candidates=sorted_candidates,
        )


def compute_phase_shares(
    delta_l1: float | None,
    delta_l2: float | None,
    delta_l3: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Compute positive normalized phase shares (L1, L2, L3) for a step event."""
    if delta_l1 is None or delta_l2 is None or delta_l3 is None:
        return None, None, None
    p1 = max(0.0, delta_l1)
    p2 = max(0.0, delta_l2)
    p3 = max(0.0, delta_l3)
    total = p1 + p2 + p3
    if total < 10.0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return round(p1 / total, 4), round(p2 / total, 4), round(p3 / total, 4)


def extract_signature_features(
    pre_samples: list[PowerSample],
    post_samples: list[PowerSample],
    event_dt: datetime | None = None,
) -> SignatureFeatures:
    """Extract electrical signature features from pre-event and post-event samples."""
    if not pre_samples or not post_samples:
        raise ValueError("pre_samples and post_samples must be non-empty")

    pre_mean_w = sum(s.total_w for s in pre_samples) / len(pre_samples)
    # Use the tail of post_samples for steady state, and max of post_samples for transient peak
    steady_slice = post_samples[1:] if len(post_samples) > 2 else post_samples
    post_steady_w = sum(s.total_w for s in steady_slice) / len(steady_slice)
    post_peak_w = max(s.total_w for s in post_samples)

    delta_p = max(0.0, post_steady_w - pre_mean_w)
    peak_overshoot = max(0.0, post_peak_w - post_steady_w)

    if len(steady_slice) >= 2:
        variance = sum((s.total_w - post_steady_w) ** 2 for s in steady_slice) / len(
            steady_slice
        )
        steady_std = math.sqrt(variance)
    else:
        steady_std = 0.0

    # Check 3-phase availability
    if all(s.has_three_phase for s in pre_samples) and all(
        s.has_three_phase for s in steady_slice
    ):
        pre_l1 = sum(s.l1_w or 0.0 for s in pre_samples) / len(pre_samples)
        pre_l2 = sum(s.l2_w or 0.0 for s in pre_samples) / len(pre_samples)
        pre_l3 = sum(s.l3_w or 0.0 for s in pre_samples) / len(pre_samples)

        post_l1 = sum(s.l1_w or 0.0 for s in steady_slice) / len(steady_slice)
        post_l2 = sum(s.l2_w or 0.0 for s in steady_slice) / len(steady_slice)
        post_l3 = sum(s.l3_w or 0.0 for s in steady_slice) / len(steady_slice)

        l1_share, l2_share, l3_share = compute_phase_shares(
            post_l1 - pre_l1, post_l2 - pre_l2, post_l3 - pre_l3
        )
    else:
        l1_share, l2_share, l3_share = None, None, None

    if event_dt is None:
        event_dt = datetime.fromtimestamp(post_samples[0].timestamp, tz=timezone.utc)

    fractional_hour = (
        event_dt.hour + event_dt.minute / 60.0 + event_dt.second / 3600.0
    )
    angle = 2.0 * math.pi * (fractional_hour / 24.0)
    hour_sin = round(math.sin(angle), 4)
    hour_cos = round(math.cos(angle), 4)

    return SignatureFeatures(
        delta_p=round(delta_p, 2),
        peak_overshoot=round(peak_overshoot, 2),
        steady_std=round(steady_std, 2),
        l1_share=l1_share,
        l2_share=l2_share,
        l3_share=l3_share,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
    )


@dataclass
class StagedStep:
    """Represents a candidate step change undergoing stabilization verification."""

    start_sample: PowerSample
    pre_slice: list[PowerSample]
    pre_mean_w: float
    initial_step_w: float
    samples: list[PowerSample] = field(default_factory=list)
    accumulated_kwh: float = 0.0
    last_sample_ts: float = 0.0


class NILMTracker:
    """Manages real-time power buffering, step-edge detection, and energy integration."""

    def __init__(
        self,
        classifier: ApplianceClassifier,
        min_power_delta: float = DEFAULT_MIN_POWER_DELTA,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        stabilization_time: float = DEFAULT_STABILIZATION_TIME,
        pre_window_size: int = 3,
        post_window_size: int = 3,
    ) -> None:
        """Initialize the NILM tracker."""
        self.classifier = classifier
        self.min_power_delta = min_power_delta
        self.confidence_threshold = confidence_threshold
        self.stabilization_time = stabilization_time
        self.pre_window_size = pre_window_size
        self.post_window_size = post_window_size

        # Known consumers mapping: consumer_id -> friendly_name
        self.consumers: dict[str, str] = dict(DEFAULT_CONSUMERS)

        # Cumulative energy (kWh) per consumer_id
        self.consumer_energy_kwh: dict[str, float] = {
            cid: 0.0 for cid in self.consumers
        }

        # Currently active or pending events (keyed by event_id)
        self.events: dict[str, ApplianceEvent] = {}

        # Sliding sample buffer for edge detection
        self._buffer: list[PowerSample] = []
        self._cooldown_samples: int = 0
        self._last_sample_ts: float | None = None
        self._staged_step: StagedStep | None = None

    def register_consumer(self, name: str, consumer_id: str | None = None) -> str:
        """Register a new consumer and initialize its energy counter if needed."""
        cid = consumer_id or slugify_consumer_name(name)
        self.consumers[cid] = name.strip()
        if cid not in self.consumer_energy_kwh:
            self.consumer_energy_kwh[cid] = 0.0
        return cid

    def get_consumer_power(self, consumer_id: str) -> float:
        """Calculate current estimated active power (W) for a consumer."""
        total_w = 0.0
        for ev in self.events.values():
            if ev.is_active and ev.assigned_consumer_id == consumer_id:
                total_w += ev.features.delta_p
        return round(total_w, 1)

    def get_pending_events(self) -> list[ApplianceEvent]:
        """Return all events awaiting user identification (latest first)."""
        pending = [
            ev for ev in self.events.values() if ev.assigned_consumer_id is None
        ]
        pending.sort(key=lambda item: item.started_at, reverse=True)
        return pending

    def dismiss_all_pending(self) -> list[ApplianceEvent]:
        """Dismiss all currently pending events by marking them as unknown."""
        dismissed: list[ApplianceEvent] = []
        for ev in list(self.get_pending_events()):
            ev.assigned_consumer_id = CONSUMER_UNKNOWN_ID
            dismissed.append(ev)
        return dismissed

    def integrate_energy(self, sample: PowerSample) -> None:
        """Integrate power over elapsed time since last sample for all active events."""
        if self._last_sample_ts is None:
            self._last_sample_ts = sample.timestamp
            return

        dt_sec = sample.timestamp - self._last_sample_ts
        self._last_sample_ts = sample.timestamp

        # Ignore negative or unreasonably long gaps (> 1 hour)
        if dt_sec <= 0.0 or dt_sec > 3600.0:
            return

        dt_hours = dt_sec / 3600.0
        for ev in self.events.values():
            if not ev.is_active:
                continue
            delta_kwh = (ev.features.delta_p * dt_hours) / 1000.0
            ev.accumulated_kwh += delta_kwh
            ev.last_integrated_ts = sample.timestamp
            if (
                ev.assigned_consumer_id is not None
                and ev.assigned_consumer_id != CONSUMER_UNKNOWN_ID
            ):
                self.consumer_energy_kwh[ev.assigned_consumer_id] = (
                    self.consumer_energy_kwh.get(ev.assigned_consumer_id, 0.0)
                    + delta_kwh
                )

    def process_sample(
        self, sample: PowerSample
    ) -> tuple[ApplianceEvent | None, list[ApplianceEvent]]:
        """Process a new power sample.

        Returns:
            (new_on_event, turned_off_events)
            where new_on_event is an ApplianceEvent if a turn-ON step was detected and stabilized.
        """
        self.integrate_energy(sample)

        # 1. Handle currently staged step change (waiting for stabilization & filtering short spikes)
        if self._staged_step is not None:
            dt_sec = sample.timestamp - self._staged_step.last_sample_ts
            if 0.0 < dt_sec <= 3600.0:
                delta_w = max(0.0, sample.total_w - self._staged_step.pre_mean_w)
                self._staged_step.accumulated_kwh += (delta_w * (dt_sec / 3600.0)) / 1000.0
            self._staged_step.last_sample_ts = sample.timestamp
            self._staged_step.samples.append(sample)

            # Check for transient spike: power dropped back near or below baseline before stabilization_time
            current_delta = sample.total_w - self._staged_step.pre_mean_w
            spike_threshold = max(
                self.min_power_delta * 0.5, self._staged_step.initial_step_w * 0.4
            )
            if current_delta < spike_threshold:
                spike_dur = sample.timestamp - self._staged_step.start_sample.timestamp
                _LOGGER.info(
                    "Ignored transient power spike (+%.1f W lasting %.1fs < %.1fs stabilization threshold)",
                    self._staged_step.initial_step_w,
                    spike_dur,
                    self.stabilization_time,
                )
                self._staged_step = None
                return None, []

            # Check if stabilization window has elapsed
            elapsed = sample.timestamp - self._staged_step.start_sample.timestamp
            if elapsed >= self.stabilization_time:
                staged = self._staged_step
                self._staged_step = None

                post_slice = (
                    staged.samples[-self.post_window_size :]
                    if len(staged.samples) >= self.post_window_size
                    else staged.samples
                )
                features = extract_signature_features(staged.pre_slice, post_slice)

                max_peak_w = max(s.total_w for s in staged.samples)
                steady_w = sum(s.total_w for s in post_slice) / len(post_slice)
                features.peak_overshoot = max(0.0, round(max_peak_w - steady_w, 2))

                self._cooldown_samples = self.post_window_size
                on_event = self._handle_turn_on(
                    staged.start_sample,
                    features,
                    initial_kwh=staged.accumulated_kwh,
                    now_ts=sample.timestamp,
                )
                return on_event, []

            # Still in stabilization window
            return None, []

        self._buffer.append(sample)

        required_len = self.pre_window_size + self.post_window_size
        if len(self._buffer) > required_len + 2:
            self._buffer.pop(0)

        if self._cooldown_samples > 0:
            self._cooldown_samples -= 1
            return None, []

        if len(self._buffer) < required_len:
            return None, []

        pre_slice = self._buffer[-required_len : -self.post_window_size]
        post_slice = self._buffer[-self.post_window_size :]

        pre_mean = sum(s.total_w for s in pre_slice) / len(pre_slice)
        post_mean = sum(s.total_w for s in post_slice) / len(post_slice)

        pre_spread = max(s.total_w for s in pre_slice) - min(s.total_w for s in pre_slice)
        step_delta = post_mean - pre_mean

        # 2. Detect Turn-ON Event
        if step_delta >= self.min_power_delta and pre_spread < max(step_delta * 0.6, 150.0):
            slice_elapsed = post_slice[-1].timestamp - post_slice[0].timestamp
            initial_kwh = 0.0
            for i in range(len(post_slice) - 1):
                dt = post_slice[i + 1].timestamp - post_slice[i].timestamp
                if 0.0 < dt <= 3600.0:
                    dw = max(0.0, post_slice[i + 1].total_w - pre_mean)
                    initial_kwh += (dw * (dt / 3600.0)) / 1000.0

            if self.stabilization_time > 0 and slice_elapsed < self.stabilization_time:
                self._staged_step = StagedStep(
                    start_sample=post_slice[0],
                    pre_slice=list(pre_slice),
                    pre_mean_w=pre_mean,
                    initial_step_w=step_delta,
                    samples=list(post_slice),
                    accumulated_kwh=initial_kwh,
                    last_sample_ts=post_slice[-1].timestamp,
                )
                return None, []

            features = extract_signature_features(pre_slice, post_slice)
            if features.delta_p >= self.min_power_delta:
                self._cooldown_samples = self.post_window_size
                on_event = self._handle_turn_on(
                    post_slice[0],
                    features,
                    initial_kwh=initial_kwh,
                    now_ts=post_slice[-1].timestamp,
                )
                return on_event, []

        # 3. Detect Turn-OFF Event
        if step_delta <= -self.min_power_delta and pre_spread < max(abs(step_delta) * 0.6, 150.0):
            drop_w = abs(step_delta)
            # Extract phase drop shares if 3-phase is available
            if all(s.has_three_phase for s in pre_slice) and all(
                s.has_three_phase for s in post_slice
            ):
                pre_l1 = sum(s.l1_w or 0.0 for s in pre_slice) / len(pre_slice)
                pre_l2 = sum(s.l2_w or 0.0 for s in pre_slice) / len(pre_slice)
                pre_l3 = sum(s.l3_w or 0.0 for s in pre_slice) / len(pre_slice)
                post_l1 = sum(s.l1_w or 0.0 for s in post_slice) / len(post_slice)
                post_l2 = sum(s.l2_w or 0.0 for s in post_slice) / len(post_slice)
                post_l3 = sum(s.l3_w or 0.0 for s in post_slice) / len(post_slice)
                drop_shares = compute_phase_shares(
                    pre_l1 - post_l1, pre_l2 - post_l2, pre_l3 - post_l3
                )
            else:
                drop_shares = (None, None, None)

            self._cooldown_samples = self.post_window_size
            off_events = self._handle_turn_off(post_slice[0].timestamp, drop_w, drop_shares)
            return None, off_events

        return None, []

    def _handle_turn_on(
        self,
        trigger_sample: PowerSample,
        features: SignatureFeatures,
        initial_kwh: float = 0.0,
        now_ts: float | None = None,
    ) -> ApplianceEvent:
        """Create an ApplianceEvent and run ML classification."""
        prediction = self.classifier.predict(features)
        event_id = f"ev_{int(trigger_sample.timestamp)}_{uuid.uuid4().hex[:6]}"
        started_iso = datetime.fromtimestamp(
            trigger_sample.timestamp, tz=timezone.utc
        ).isoformat()

        # Check if confidence exceeds threshold for automatic assignment
        if (
            prediction.predicted_consumer_id is not None
            and prediction.confidence >= self.confidence_threshold
        ):
            assigned_id: str | None = prediction.predicted_consumer_id
        else:
            assigned_id = None

        event = ApplianceEvent(
            event_id=event_id,
            started_at=trigger_sample.timestamp,
            started_iso=started_iso,
            features=features,
            predicted_consumer_id=prediction.predicted_consumer_id,
            confidence=prediction.confidence,
            candidates=prediction.candidates,
            assigned_consumer_id=assigned_id,
            is_active=True,
            accumulated_kwh=initial_kwh,
            last_integrated_ts=now_ts if now_ts is not None else trigger_sample.timestamp,
        )
        self.events[event_id] = event

        if assigned_id is not None and assigned_id != CONSUMER_UNKNOWN_ID and initial_kwh > 0.0:
            self.consumer_energy_kwh[assigned_id] = (
                self.consumer_energy_kwh.get(assigned_id, 0.0) + initial_kwh
            )

        self._prune_old_events()
        return event

    def _handle_turn_off(
        self,
        timestamp: float,
        drop_w: float,
        drop_shares: tuple[float | None, float | None, float | None],
    ) -> list[ApplianceEvent]:
        """Match a negative power step against currently active appliance events."""
        active_events = [ev for ev in self.events.values() if ev.is_active]
        if not active_events:
            return []

        # Score each active event by how closely its delta_p and phase shares match drop_w
        best_event: ApplianceEvent | None = None
        best_diff = float("inf")

        for ev in active_events:
            rel_diff = abs(ev.features.delta_p - drop_w) / max(
                ev.features.delta_p, drop_w, 100.0
            )
            phase_penalty = 0.0
            if (
                drop_shares[0] is not None
                and ev.features.l1_share is not None
                and ev.features.l2_share is not None
                and ev.features.l3_share is not None
            ):
                phase_penalty = (
                    abs(ev.features.l1_share - (drop_shares[0] or 0.0))
                    + abs(ev.features.l2_share - (drop_shares[1] or 0.0))
                    + abs(ev.features.l3_share - (drop_shares[2] or 0.0))
                ) * 0.5

            score = rel_diff + phase_penalty
            if score < best_diff:
                best_diff = score
                best_event = ev

        # Accept match if reasonably close (< 55% relative mismatch or only 1 active load)
        if best_event is not None and (best_diff <= 0.55 or len(active_events) == 1):
            best_event.is_active = False
            best_event.ended_at = timestamp
            return [best_event]

        # Also check if the power drop matches the sum of 2 active appliances turning off together
        if len(active_events) >= 2:
            for i in range(len(active_events)):
                for j in range(i + 1, len(active_events)):
                    combined_w = (
                        active_events[i].features.delta_p
                        + active_events[j].features.delta_p
                    )
                    rel_diff = abs(combined_w - drop_w) / max(combined_w, drop_w, 100.0)
                    if rel_diff <= 0.35:
                        active_events[i].is_active = False
                        active_events[i].ended_at = timestamp
                        active_events[j].is_active = False
                        active_events[j].ended_at = timestamp
                        return [active_events[i], active_events[j]]

        return []

    def label_event(
        self,
        consumer_id: str,
        event_id: str | None = None,
    ) -> ApplianceEvent | None:
        """Assign a user-selected consumer label to a pending (or specified) event.

        Also trains the ML classifier with the event's signature (unless 'unknown')
        and retroactively credits the event's accumulated energy to the consumer.
        """
        target_event: ApplianceEvent | None = None
        if event_id and event_id in self.events:
            target_event = self.events[event_id]
        else:
            pending = self.get_pending_events()
            if pending:
                target_event = pending[0]

        if target_event is None:
            return None

        previous_consumer = target_event.assigned_consumer_id
        # If reassigning an already-assigned event, subtract previously credited energy
        if (
            previous_consumer is not None
            and previous_consumer != CONSUMER_UNKNOWN_ID
            and previous_consumer in self.consumer_energy_kwh
        ):
            self.consumer_energy_kwh[previous_consumer] = max(
                0.0,
                self.consumer_energy_kwh[previous_consumer]
                - target_event.accumulated_kwh,
            )

        target_event.assigned_consumer_id = consumer_id

        if consumer_id != CONSUMER_UNKNOWN_ID:
            if consumer_id not in self.consumer_energy_kwh:
                self.consumer_energy_kwh[consumer_id] = 0.0
            self.consumer_energy_kwh[consumer_id] += target_event.accumulated_kwh
            self.classifier.add_training_sample(
                consumer_id=consumer_id,
                features=target_event.features,
                created_at=target_event.started_iso,
            )

        return target_event

    def _prune_old_events(self, max_completed_events: int = 50) -> None:
        """Keep memory bounded by pruning old completed & labeled events."""
        completed_labeled = [
            ev
            for ev in self.events.values()
            if not ev.is_active and ev.assigned_consumer_id is not None
        ]
        if len(completed_labeled) > max_completed_events:
            completed_labeled.sort(key=lambda item: item.started_at)
            for ev in completed_labeled[: len(completed_labeled) - max_completed_events]:
                self.events.pop(ev.event_id, None)
