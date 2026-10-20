"""Constants for the HomeWizard NILM Appliance Disaggregator integration."""

from __future__ import annotations

DOMAIN = "homewizard_nilm"

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.ml_store"

# Configuration keys
CONF_POWER_SENSOR = "power_sensor"
CONF_POWER_L1_SENSOR = "power_l1_sensor"
CONF_POWER_L2_SENSOR = "power_l2_sensor"
CONF_POWER_L3_SENSOR = "power_l3_sensor"
CONF_NOTIFY_SERVICE = "notify_service"
CONF_MIN_POWER_DELTA = "min_power_delta"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"
CONF_STABILIZATION_TIME = "stabilization_time"

# Defaults
DEFAULT_MIN_POWER_DELTA = 80.0  # Watts
DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # 75% confidence required for auto-classification
DEFAULT_STABILIZATION_TIME = 15.0  # Seconds to wait for power to stabilize and ignore spikes

# Special label for unidentified events
CONSUMER_UNKNOWN_ID = "unknown"
CONSUMER_UNKNOWN_NAME = "Unknown"

# Initial pre-configured consumers requested by the user
DEFAULT_CONSUMERS: dict[str, str] = {
    "boiler": "Boiler",
    "stove": "Stove",
    "oven": "Oven",
    "washer": "Washer",
    "dryer": "Dryer",
    "dishwasher": "Dishwasher",
    "infrared_heating": "Infrared heating",
    "floor_heating_bathroom_1": "Floor heating Bathroom 1",
    "floor_heating_bathroom_2": "Floor heating Bathroom 2",
    "floor_heating_elsas_bedroom": "Floor heating Elsa's bedroom",
    "floor_heating_master_bedroom": "Floor heating Master bedroom",
}

# Swedish friendly display names
SWEDISH_DEFAULT_NAMES: dict[str, str] = {
    "boiler": "Varmvattenberedare",
    "stove": "Spis",
    "oven": "Ugn",
    "washer": "Tvättmaskin",
    "dryer": "Torktumlare",
    "dishwasher": "Diskmaskin",
    "infrared_heating": "Infravärme",
    "floor_heating_bathroom_1": "Golvvärme Badrum 1",
    "floor_heating_bathroom_2": "Golvvärme Badrum 2",
    "floor_heating_elsas_bedroom": "Golvvärme Elsas sovrum",
    "floor_heating_master_bedroom": "Golvvärme Sovrum",
}

# Services
SERVICE_LABEL_EVENT = "label_event"
SERVICE_ADD_CONSUMER = "add_consumer"
SERVICE_RESET_MODEL = "reset_model"
SERVICE_DISMISS_ALL_PENDING = "dismiss_all_pending"
SERVICE_START_CONSUMER = "start_consumer"
SERVICE_STOP_CONSUMER = "stop_consumer"

# Dispatcher signals
SIGNAL_NEW_CONSUMER_ADDED = f"{DOMAIN}_new_consumer_added"
SIGNAL_STATE_UPDATED = f"{DOMAIN}_state_updated"

# Mobile App Notification Action Prefixes
ACTION_PREFIX_SELECT = "HW_NILM_SELECT_"
ACTION_PREFIX_UNKNOWN = "HW_NILM_UNKNOWN_"
ACTION_PREFIX_ADD_NEW = "HW_NILM_ADD_NEW_"
