"""Coordinator and state manager for HomeWizard NILM Appliance Disaggregator."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.storage import Store

from .const import (
    ACTION_PREFIX_ADD_NEW,
    ACTION_PREFIX_SELECT,
    ACTION_PREFIX_UNKNOWN,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_MIN_POWER_DELTA,
    CONF_NOTIFY_SERVICE,
    CONF_POWER_L1_SENSOR,
    CONF_POWER_L2_SENSOR,
    CONF_POWER_L3_SENSOR,
    CONF_POWER_SENSOR,
    CONF_STABILIZATION_TIME,
    CONSUMER_UNKNOWN_ID,
    CONSUMER_UNKNOWN_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONSUMERS,
    DEFAULT_MIN_POWER_DELTA,
    DEFAULT_STABILIZATION_TIME,
    DOMAIN,
    SIGNAL_NEW_CONSUMER_ADDED,
    SIGNAL_STATE_UPDATED,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from .ml_engine import (
    ApplianceClassifier,
    ApplianceEvent,
    NILMTracker,
    PowerSample,
    slugify_consumer_name,
)

_LOGGER = logging.getLogger(__name__)


class HomeWizardNILMCoordinator:
    """Manages HomeWizard sensor listeners, ML disaggregation, notifications, and storage."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        self.hass = hass
        self.entry = entry
        self.store = Store[dict[str, Any]](
            hass, STORAGE_VERSION, f"{STORAGE_KEY}.{entry.entry_id}"
        )

        config = {**entry.data, **entry.options}
        self.power_sensor: str = config[CONF_POWER_SENSOR]
        self.power_l1_sensor: str | None = config.get(CONF_POWER_L1_SENSOR) or None
        self.power_l2_sensor: str | None = config.get(CONF_POWER_L2_SENSOR) or None
        self.power_l3_sensor: str | None = config.get(CONF_POWER_L3_SENSOR) or None
        self.notify_service: str | None = config.get(CONF_NOTIFY_SERVICE) or None

        min_delta = float(config.get(CONF_MIN_POWER_DELTA, DEFAULT_MIN_POWER_DELTA))
        conf_threshold = float(
            config.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
        )
        stabilization_time = float(
            config.get(CONF_STABILIZATION_TIME, DEFAULT_STABILIZATION_TIME)
        )

        self.classifier = ApplianceClassifier()
        self.tracker = NILMTracker(
            classifier=self.classifier,
            min_power_delta=min_delta,
            confidence_threshold=conf_threshold,
            stabilization_time=stabilization_time,
        )

        # Staging text value entered in the "New Consumer Name" Text entity
        self.staged_new_consumer_name: str = ""
        self.last_event: ApplianceEvent | None = None

        self._unsub_state_listener: Any = None
        self._unsub_mobile_action_listener: Any = None

    async def async_setup(self) -> None:
        """Load stored ML samples and consumers, and subscribe to sensor events."""
        await self._async_load_storage()

        # Subscribe to primary power sensor changes
        self._unsub_state_listener = async_track_state_change_event(
            self.hass,
            [self.power_sensor],
            self._async_on_power_state_change,
        )

        # Subscribe to Companion App actionable notification responses
        self._unsub_mobile_action_listener = self.hass.bus.async_listen(
            "mobile_app_notification_action",
            self._async_on_mobile_notification_action,
        )

    async def async_unload(self) -> None:
        """Clean up listeners and persist state."""
        if self._unsub_state_listener:
            self._unsub_state_listener()
            self._unsub_state_listener = None

        if self._unsub_mobile_action_listener:
            self._unsub_mobile_action_listener()
            self._unsub_mobile_action_listener = None

        await self.async_save_storage()

    async def _async_load_storage(self) -> None:
        """Load persisted consumers, cumulative energy, and ML training samples."""
        data = await self.store.async_load()
        if not data:
            return

        stored_consumers = data.get("consumers", {})
        if isinstance(stored_consumers, dict):
            for cid, name in stored_consumers.items():
                self.tracker.register_consumer(str(name), str(cid))

        stored_energy = data.get("consumer_energy_kwh", {})
        if isinstance(stored_energy, dict):
            for cid, val in stored_energy.items():
                try:
                    self.tracker.consumer_energy_kwh[str(cid)] = float(val)
                except (TypeError, ValueError):
                    continue

        raw_samples = data.get("training_samples", [])
        if isinstance(raw_samples, list):
            self.classifier.load_samples(raw_samples)

        _LOGGER.info(
            "Loaded HomeWizard NILM store: %d consumers, %d training signatures",
            len(self.tracker.consumers),
            len(self.classifier.samples),
        )

    async def async_save_storage(self) -> None:
        """Persist consumers, cumulative energy, and ML training samples."""
        payload = {
            "consumers": self.tracker.consumers,
            "consumer_energy_kwh": {
                cid: round(val, 6)
                for cid, val in self.tracker.consumer_energy_kwh.items()
            },
            "training_samples": self.classifier.dump_samples(),
        }
        await self.store.async_save(payload)

    def _read_float_state(self, entity_id: str | None) -> float | None:
        """Read a numeric float value from a Home Assistant state object."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            return None

    @callback
    def _async_on_power_state_change(self, event: Event) -> None:
        """Handle new power reading from the HomeWizard active power sensor."""
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        try:
            total_w = float(new_state.state)
        except (TypeError, ValueError):
            return

        l1_w = self._read_float_state(self.power_l1_sensor)
        l2_w = self._read_float_state(self.power_l2_sensor)
        l3_w = self._read_float_state(self.power_l3_sensor)

        ts = new_state.last_updated.timestamp() if new_state.last_updated else (
            datetime.now(timezone.utc).timestamp()
        )

        sample = PowerSample(
            timestamp=ts,
            total_w=total_w,
            l1_w=l1_w,
            l2_w=l2_w,
            l3_w=l3_w,
        )

        new_on_event, off_events = self.tracker.process_sample(sample)
        if new_on_event is not None:
            self.last_event = new_on_event
            if new_on_event.assigned_consumer_id is None:
                # Uncertain or cold-start signature: notify user to identify consumer
                self.hass.async_create_task(
                    self._async_prompt_user_for_event(new_on_event)
                )
            else:
                cname = self.tracker.consumers.get(
                    new_on_event.assigned_consumer_id,
                    new_on_event.assigned_consumer_id,
                )
                _LOGGER.info(
                    "Auto-identified consumer '%s' (%.1f W, confidence %.1f%%)",
                    cname,
                    new_on_event.features.delta_p,
                    new_on_event.confidence * 100.0,
                )

        if off_events:
            self.hass.async_create_task(self.async_save_storage())

        async_dispatcher_send(
            self.hass, f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}"
        )

    async def _async_prompt_user_for_event(self, event: ApplianceEvent) -> None:
        """Send a Persistent Notification and optional Mobile App notification."""
        delta_w = round(event.features.delta_p)
        phase_str = event.features.dominant_phase
        started_time = event.started_time_str()
        duration_str = event.duration_str()
        energy_str = event.consumed_energy_str()

        # Detailed per-phase breakdown if 3-phase data is present
        phase_breakdown = ""
        if (
            event.features.l1_share is not None
            and event.features.l2_share is not None
            and event.features.l3_share is not None
        ):
            p1 = round(delta_w * event.features.l1_share)
            p2 = round(delta_w * event.features.l2_share)
            p3 = round(delta_w * event.features.l3_share)
            phase_breakdown = f"\n- **Phase Breakdown**: L1: `{p1} W` | L2: `{p2} W` | L3: `{p3} W`"

        if event.predicted_consumer_id:
            guess_name = self.tracker.consumers.get(
                event.predicted_consumer_id, event.predicted_consumer_id
            )
            guess_line = (
                f"💡 **Suggested Consumer**: **{guess_name}** "
                f"({event.confidence * 100:.0f}% confidence — below "
                f"{self.tracker.confidence_threshold * 100:.0f}% auto-classification threshold)."
            )
        else:
            guess_line = (
                "💡 **Cold Start**: No matching appliance signature learned yet."
            )

        consumer_names = ", ".join(self.tracker.consumers.values())
        notification_id = f"{DOMAIN}_{event.event_id}"

        message = (
            f"### ⚡ New Power Consumer Detected\n\n"
            f"- **Turned ON at**: `{started_time}` (`{duration_str} ago`)\n"
            f"- **Stabilized Power**: **+{delta_w} W** (Phase: `{phase_str}`){phase_breakdown}\n"
            f"- **Energy Consumed So Far**: **{energy_str}**\n\n"
            f"{guess_line}\n\n"
            f"### How to identify this consumer:\n"
            f"1. **Select Existing Consumer or '{CONSUMER_UNKNOWN_NAME}'**: Use the "
            f"**HomeWizard NILM Identify Consumer** dropdown (`select.homewizard_nilm_identify_consumer`).\n"
            f"2. **Add a New Consumer**: Enter the name in **HomeWizard NILM New Consumer Name** "
            f"(`text.homewizard_nilm_new_consumer_name`) and press **Add New Consumer & Assign** "
            f"(`button.homewizard_nilm_confirm_new_consumer`), or reply directly to the mobile notification.\n\n"
            f"*Available consumers*: {consumer_names}, or **{CONSUMER_UNKNOWN_NAME}**."
        )

        persistent_notification.async_create(
            self.hass,
            message=message,
            title=f"⚡ Identify Appliance: +{delta_w} W ({duration_str} ago)",
            notification_id=notification_id,
        )

        # Also send Mobile App actionable notification if configured
        if self.notify_service:
            await self._async_send_mobile_notification(event)

    async def _async_send_mobile_notification(self, event: ApplianceEvent) -> None:
        """Send an actionable notification via notify.<service>."""
        if not self.notify_service:
            return

        service_parts = self.notify_service.split(".", 1)
        domain = service_parts[0] if len(service_parts) == 2 else "notify"
        service_name = (
            service_parts[1] if len(service_parts) == 2 else self.notify_service
        )

        # Build quick-action buttons: up to 2 top candidate consumers + Unknown + Add/Reply New Consumer
        actions: list[dict[str, Any]] = []
        candidate_ids = [cid for cid, _ in event.candidates[:2]]

        # If cold start (no candidates yet), offer the first 2 unlearned consumers as quick buttons
        if not candidate_ids:
            learned_ids = {s.consumer_id for s in self.classifier.samples}
            unlearned = [
                cid for cid in self.tracker.consumers if cid not in learned_ids
            ]
            candidate_ids = unlearned[:2]

        for cid in candidate_ids:
            cname = self.tracker.consumers.get(cid, cid)
            actions.append(
                {
                    "action": f"{ACTION_PREFIX_SELECT}{event.event_id}__{cid}",
                    "title": f"It's {cname}",
                }
            )

        actions.append(
            {
                "action": f"{ACTION_PREFIX_UNKNOWN}{event.event_id}",
                "title": CONSUMER_UNKNOWN_NAME,
            }
        )
        actions.append(
            {
                "action": f"{ACTION_PREFIX_ADD_NEW}{event.event_id}",
                "title": "Other / Add New...",
                "behavior": "textInput",
                "textInputButtonTitle": "Save & Assign",
                "textInputPlaceholder": "Enter consumer name (new or existing)",
            }
        )

        delta_w = round(event.features.delta_p)
        phase_str = event.features.dominant_phase
        started_time = event.started_time_str()
        duration_str = event.duration_str()
        energy_str = event.consumed_energy_str()

        if event.predicted_consumer_id:
            guess_name = self.tracker.consumers.get(
                event.predicted_consumer_id, event.predicted_consumer_id
            )
            body = (
                f"Started at {started_time} ({duration_str} ago). "
                f"Stabilized draw: +{delta_w}W ({phase_str}), consumed {energy_str}. "
                f"Best guess: {guess_name} ({event.confidence * 100:.0f}%). Which consumer is this?"
            )
        else:
            body = (
                f"Started at {started_time} ({duration_str} ago). "
                f"Stabilized draw: +{delta_w}W ({phase_str}), consumed {energy_str}. "
                f"Which consumer was turned on?"
            )

        try:
            await self.hass.services.async_call(
                domain,
                service_name,
                {
                    "title": f"⚡ Identify Appliance: +{delta_w}W ({duration_str} ago)",
                    "message": body,
                    "data": {
                        "tag": f"{DOMAIN}_{event.event_id}",
                        "actions": actions,
                    },
                },
                blocking=False,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning(
                "Failed to send mobile notification via %s: %s",
                self.notify_service,
                err,
            )

    @callback
    def _async_on_mobile_notification_action(self, event: Event) -> None:
        """Handle actionable notification responses from the Home Assistant Companion App."""
        action: str = event.data.get("action", "")
        if not action:
            return

        if action.startswith(ACTION_PREFIX_SELECT):
            payload = action[len(ACTION_PREFIX_SELECT) :]
            if "__" in payload:
                event_id, consumer_id = payload.split("__", 1)
                self.hass.async_create_task(
                    self.async_label_event(consumer_id=consumer_id, event_id=event_id)
                )

        elif action.startswith(ACTION_PREFIX_UNKNOWN):
            event_id = action[len(ACTION_PREFIX_UNKNOWN) :]
            self.hass.async_create_task(
                self.async_label_event(
                    consumer_id=CONSUMER_UNKNOWN_ID, event_id=event_id
                )
            )

        elif action.startswith(ACTION_PREFIX_ADD_NEW):
            event_id = action[len(ACTION_PREFIX_ADD_NEW) :]
            reply_text: str = (
                event.data.get("reply_text")
                or event.data.get("textInput")
                or ""
            ).strip()
            if reply_text:
                self.hass.async_create_task(
                    self.async_add_consumer_and_label(
                        name=reply_text, event_id=event_id
                    )
                )

    async def async_label_event(
        self, consumer_id: str, event_id: str | None = None
    ) -> ApplianceEvent | None:
        """Label a pending (or specified) event with a consumer ID or 'unknown'."""
        # Allow matching by friendly name as well as slug ID
        resolved_id = self._resolve_consumer_id(consumer_id)
        labeled_event = self.tracker.label_event(
            consumer_id=resolved_id, event_id=event_id
        )
        if labeled_event is None:
            _LOGGER.debug("No matching pending event found to label.")
            return None

        # Dismiss the persistent notification for this event
        persistent_notification.async_dismiss(
            self.hass, notification_id=f"{DOMAIN}_{labeled_event.event_id}"
        )

        await self.async_save_storage()
        async_dispatcher_send(
            self.hass, f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}"
        )
        return labeled_event

    async def async_add_consumer_and_label(
        self, name: str, event_id: str | None = None
    ) -> str:
        """Dynamically add a new consumer (creating its sensors) and optionally label the current event."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Consumer name cannot be empty")

        if clean_name.lower() == CONSUMER_UNKNOWN_ID:
            await self.async_label_event(
                consumer_id=CONSUMER_UNKNOWN_ID, event_id=event_id
            )
            return CONSUMER_UNKNOWN_ID

        cid = slugify_consumer_name(clean_name)
        is_new = cid not in self.tracker.consumers
        self.tracker.register_consumer(clean_name, cid)

        if is_new:
            _LOGGER.info("Added new consumer '%s' (id: %s)", clean_name, cid)
            async_dispatcher_send(
                self.hass,
                f"{SIGNAL_NEW_CONSUMER_ADDED}_{self.entry.entry_id}",
                cid,
            )

        # If there is a pending event (or explicit event_id), assign it to this new consumer!
        if event_id or self.tracker.get_pending_events():
            await self.async_label_event(consumer_id=cid, event_id=event_id)
        else:
            await self.async_save_storage()
            async_dispatcher_send(
                self.hass, f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}"
            )

        return cid

    async def async_reset_model(self) -> None:
        """Clear learned ML training samples while preserving consumers and energy totals."""
        self.classifier.samples.clear()
        await self.async_save_storage()
        async_dispatcher_send(
            self.hass, f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}"
        )

    def _resolve_consumer_id(self, identifier: str) -> str:
        """Resolve either a consumer_id or a friendly name to a canonical consumer_id."""
        cleaned = identifier.strip()
        if cleaned.lower() in (CONSUMER_UNKNOWN_ID, CONSUMER_UNKNOWN_NAME.lower()):
            return CONSUMER_UNKNOWN_ID
        if cleaned in self.tracker.consumers:
            return cleaned
        for cid, fname in self.tracker.consumers.items():
            if fname.lower() == cleaned.lower():
                return cid
        slug = slugify_consumer_name(cleaned)
        if slug in self.tracker.consumers:
            return slug
        return slug
