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

PROMPT_PLACEHOLDER = "-- Select Consumer for Pending Event --"


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
    _attr_icon = "mdi:FormatListChecks"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the select entity."""
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_identify_consumer"
        self._attr_name = "Identify Consumer"
        self._attr_icon = "mdi:format-list-checks"

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

    @property
    def options(self) -> list[str]:
        """Return the list of selectable consumers plus 'Unknown'."""
        consumer_names = list(self.coordinator.tracker.consumers.values())
        return [PROMPT_PLACEHOLDER, *consumer_names, CONSUMER_UNKNOWN_NAME]

    @property
    def current_option(self) -> str:
        """Always display the placeholder when awaiting selection."""
        return PROMPT_PLACEHOLDER

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Show details about the pending event that will be labeled upon selection."""
        pending = self.coordinator.tracker.get_pending_events()
        if not pending:
            return {"pending_event": None, "status": "No pending events"}
        ev = pending[0]
        return {
            "pending_event": ev.to_summary_dict(self.coordinator.tracker.consumers),
            "status": f"Awaiting selection for +{round(ev.features.delta_p)}W event",
        }

    async def async_select_option(self, option: str) -> None:
        """Label the current pending event with the user's selected consumer or 'Unknown'."""
        if option == PROMPT_PLACEHOLDER:
            return

        if option == CONSUMER_UNKNOWN_NAME:
            await self.coordinator.async_label_event(consumer_id=CONSUMER_UNKNOWN_ID)
        else:
            await self.coordinator.async_label_event(consumer_id=option)

        self.async_write_ha_state()
