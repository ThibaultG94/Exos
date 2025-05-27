"""
Module de configuration
"""
import json
import os

def charger_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"difficulte": "normal"}