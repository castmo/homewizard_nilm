"""Image platform for HomeWizard NILM appliance power timeline diagrams."""

from __future__ import annotations

from datetime import datetime, timezone

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SIGNAL_STATE_UPDATED
from .coordinator import HomeWizardNILMCoordinator
from .ml_engine import generate_idle_diagram_svg
from .sensor import get_hub_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the HomeWizard NILM Event Diagram Image entity."""
    coordinator: HomeWizardNILMCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([NILMEventDiagramImage(coordinator, entry)])


class NILMEventDiagramImage(ImageEntity):
    """Image entity serving the SVG power timeline diagram for the latest pending event."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_content_type = "image/svg+xml"
    _attr_icon = "mdi:chart-timeline-variant"

    def __init__(
        self, coordinator: HomeWizardNILMCoordinator, entry: ConfigEntry
    ) -> None:
        """Initialize the image entity."""
        super().__init__(coordinator.hass)
        self.coordinator = coordinator
        self.entry = entry
        self._attr_device_info = get_hub_device_info(entry)
        self._attr_unique_id = f"{entry.entry_id}_event_diagram"
        self._attr_name = "Event Diagram"
        self._attr_image_last_updated = datetime.now(timezone.utc)
        self.coordinator.diagram_image_entity = self

    async def async_added_to_hass(self) -> None:
        """Subscribe to coordinator state updates."""
        await super().async_added_to_hass()
        self.coordinator.diagram_image_entity = self
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_STATE_UPDATED}_{self.entry.entry_id}",
                self._async_on_state_updated,
            )
        )

    def _async_on_state_updated(self) -> None:
        """Update last updated timestamp and refresh HA state."""
        self._attr_image_last_updated = datetime.now(timezone.utc)
        self.async_write_ha_state()

    async def async_image(self) -> bytes | None:
        """Return the dynamic SVG bytes representing the current appliance event or idle state."""
        pending = self.coordinator.tracker.get_pending_events()
        if pending:
            svg_str = pending[0].generate_diagram_svg()
        else:
            svg_str = generate_idle_diagram_svg()
        return svg_str.encode("utf-8")
