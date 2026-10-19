"""Select platform for HomeWizard NILM user identification prompts."""

from __future__ import annotations

from typing import Any

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONSUMER_UNKNOWN_ID,
    CONSUMER_UNKNOWN_NAME,
    DOMAIN,
    SIGNAL_NEW_CONSUMER_ADDED,
    SIGNAL_STATE_UPDATED,
)
from .coordinator import HomeWizardNILMCoordinator
from .sensor import get_hub_device_info

PROMPT_IDLE = "✅ Inga väntande förbrukare"
OPTION_UNKNOWN = "❓ Okänd förbrukare"
OPTION_DISMISS_ALL = "🗑️ Avfärda alla väntande"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the HomeWizard NILM consumer identification Select entity."""
    coordinator: HomeWizardNILMCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([NILMIdentifyConsumerSelect(coordinator, entry)])


class NILMIdentifyConsumerSelect(SelectEntity):
    """Dropdown entity allowing the user to identify which consumer was turned on."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_icon = "mdi:format-list-checks"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the select entity."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_identify_consumer"
        self._attr_name = "Identify Consumer"

    async def async_added_to_hass(self) -> None:
        """Subscribe to state and new consumer signals."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}",
                self.async_write_ha_state,
            )
        )
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_NEW_CONSUMER_ADDED}_{self.entry.entry_id}",
                self._async_on_consumer_added,
            )
        )

    @callback
    def _async_on_consumer_added(self, _consumer_id: str) -> None:
        """Refresh options when a new consumer is dynamically added."""
        self.async_write_ha_state()

    def _get_placeholder(self) -> str:
        """Return the contextual Swedish placeholder for the dropdown."""
        pending = self.coordinator.tracker.get_pending_events()
        if not pending:
            return PROMPT_IDLE
        ev = pending[0]
        w_str = ev.wattage_str()
        time_part = ev.started_time_str()[:5]
        phase = ev.features.dominant_phase
        q_count = len(pending)
        if q_count > 1:
            return f"-- Välj för {w_str} (kl. {time_part}, {phase}) [{q_count} i kö] --"
        return f"-- Välj för {w_str} (kl. {time_part}, {phase}) --"

    @property
    def options(self) -> list[str]:
        """Return the dynamic placeholder, prioritized consumers, Unknown, and Dismiss All."""
        placeholder = self._get_placeholder()
        consumer_names = list(self.coordinator.tracker.consumers.values())

        # If a pending event has a predicted consumer with reasonable confidence, prioritize it
        pending = self.coordinator.tracker.get_pending_events()
        if pending and pending[0].predicted_consumer_id and pending[0].confidence >= 0.15:
            pred_id = pending[0].predicted_consumer_id
            pred_name = self.coordinator.tracker.consumers.get(pred_id, pred_id)
            if pred_name in consumer_names:
                consumer_names.remove(pred_name)
                consumer_names.insert(0, pred_name)

        opts = [placeholder, *consumer_names, OPTION_UNKNOWN]
        if pending:
            opts.append(OPTION_DISMISS_ALL)
        return opts

    @property
    def current_option(self) -> str:
        """Display the contextual prompt placeholder."""
        return self._get_placeholder()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Show clean, human-readable Swedish details about the pending event in the more-info popup."""
        pending = self.coordinator.tracker.get_pending_events()
        if not pending:
            return {
                "status": "Inga väntande förbrukare",
                "väntande_i_kö": 0,
            }
        ev = pending[0]
        consumers = self.coordinator.tracker.consumers
        has_ended = not ev.is_active and ev.ended_at is not None
        if has_ended:
            time_str = f"Kl. {ev.started_time_str()} – {ev.ended_time_str()} ({ev.duration_str_sv()})"
        else:
            time_str = f"Startade kl. {ev.started_time_str()} (pågår i {ev.duration_str_sv()})"

        if ev.predicted_consumer_id and ev.confidence >= 0.15:
            pred_name = consumers.get(ev.predicted_consumer_id, ev.predicted_consumer_id)
            pred_str = f"{pred_name} ({round(ev.confidence * 100)} %)"
        else:
            pred_str = "Ingen matchning ännu"

        return {
            "effekt": f"{ev.wattage_str()} ({ev.features.dominant_phase})",
            "tidsförlopp": time_str,
            "status": ev.status_str_sv(),
            "förbrukad_energi": ev.consumed_energy_str_sv(),
            "ml_förslag": pred_str,
            "väntande_i_kö": len(pending),
        }

    async def async_select_option(self, option: str) -> None:
        """Label or dismiss the pending event based on user dropdown selection."""
        placeholder = self._get_placeholder()
        if option == placeholder or option.startswith("--") or option == PROMPT_IDLE:
            return

        if option in (OPTION_DISMISS_ALL, "Avfärda alla väntande"):
            await self.coordinator.async_dismiss_all_pending()
        elif option in (OPTION_UNKNOWN, CONSUMER_UNKNOWN_NAME, "Okänd förbrukare"):
            await self.coordinator.async_label_event(consumer_id=CONSUMER_UNKNOWN_ID)
        else:
            await self.coordinator.async_label_event(consumer_id=option)

        self.async_write_ha_state()
