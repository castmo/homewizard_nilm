"""Config flow for HomeWizard NILM Appliance Disaggregator integration."""

from __future__ import annotations

from typing import Any
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_CONFIDENCE_THRESHOLD,
    CONF_MIN_POWER_DELTA,
    CONF_NOTIFY_SERVICE,
    CONF_POWER_L1_SENSOR,
    CONF_POWER_L2_SENSOR,
    CONF_POWER_L3_SENSOR,
    CONF_POWER_SENSOR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_POWER_DELTA,
    DOMAIN,
)


def _build_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the configuration schema for user setup and options flow."""
    power_entity_selector = selector.EntitySelector(
        selector.EntitySelectorConfig(
            domain="sensor",
            device_class=SensorDeviceClass.POWER,
        )
    )
    optional_power_selector = selector.EntitySelector(
        selector.EntitySelectorConfig(
            domain="sensor",
            device_class=SensorDeviceClass.POWER,
            multiple=False,
        )
    )

    schema_dict: dict[Any, Any] = {
        vol.Required(
            CONF_POWER_SENSOR,
            default=defaults.get(CONF_POWER_SENSOR, vol.UNDEFINED),
        ): power_entity_selector,
        vol.Optional(
            CONF_POWER_L1_SENSOR,
            description={"suggested_value": defaults.get(CONF_POWER_L1_SENSOR)},
        ): optional_power_selector,
        vol.Optional(
            CONF_POWER_L2_SENSOR,
            description={"suggested_value": defaults.get(CONF_POWER_L2_SENSOR)},
        ): optional_power_selector,
        vol.Optional(
            CONF_POWER_L3_SENSOR,
            description={"suggested_value": defaults.get(CONF_POWER_L3_SENSOR)},
        ): optional_power_selector,
        vol.Optional(
            CONF_NOTIFY_SERVICE,
            default=defaults.get(CONF_NOTIFY_SERVICE, ""),
        ): selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)
        ),
        vol.Optional(
            CONF_MIN_POWER_DELTA,
            default=defaults.get(CONF_MIN_POWER_DELTA, DEFAULT_MIN_POWER_DELTA),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=20.0,
                max=2000.0,
                step=10.0,
                unit_of_measurement="W",
                mode=selector.NumberSelectorMode.BOX,
            )
        ),
        vol.Optional(
            CONF_CONFIDENCE_THRESHOLD,
            default=defaults.get(
                CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
            ),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.30,
                max=0.99,
                step=0.05,
                mode=selector.NumberSelectorMode.SLIDER,
            )
        ),
    }
    return vol.Schema(schema_dict)


class HomeWizardNILMConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HomeWizard NILM Appliance Disaggregator."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is not None:
            await self.async_set_unique_id(user_input[CONF_POWER_SENSOR])
            self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title=f"HomeWizard NILM ({user_input[CONF_POWER_SENSOR]})",
                data=user_input,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema({}),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the options flow handler."""
        return HomeWizardNILMOptionsFlow(config_entry)


class HomeWizardNILMOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for HomeWizard NILM."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = {**self._config_entry.data, **self._config_entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(current),
        )
