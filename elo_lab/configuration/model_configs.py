"""
Elo model configurations.

Defines the parameter sets and adjustment pipelines used to
instantiate different Elo model variants.
"""

# Default home-field adjustment value
from ..adjustments import DEFAULT_HFA


MODEL_CONFIGS = {

    "BASE": {
        "k": 20,
        "pregame_pipeline": [],
        "postgame_pipeline": [],
        "adjustments": {
            "home_field": {
                "enabled": False,
                "value": DEFAULT_HFA,
            },
            "margin_of_victory": {
                "enabled": False,
                "scale": 1.0,
            },
        },
    },

    "MOV": {
        "k": 20,
        "pregame_pipeline": [],
        "postgame_pipeline": ["margin_of_victory"],
        "adjustments": {
            "home_field": {
                "enabled": False,
                "value": DEFAULT_HFA,
            },
            "margin_of_victory": {
                "enabled": True,
                "scale": 1.0,
            },
        },
    },

    "HFA": {
        "k": 20,
        "pregame_pipeline": ["home_field"],
        "postgame_pipeline": [],
        "adjustments": {
            "home_field": {
                "enabled": True,
                "value": DEFAULT_HFA,
            },
            "margin_of_victory": {
                "enabled": False,
                "scale": 1.0,
            },
        },
    },

    "MOV_HFA": {
        "k": 20,
        "pregame_pipeline": ["home_field"],
        "postgame_pipeline": ["margin_of_victory"],
        "adjustments": {
            "home_field": {
                "enabled": True,
                "value": DEFAULT_HFA,
            },
            "margin_of_victory": {
                "enabled": True,
                "scale": 1.0,
            },
        },
    },
}


AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())