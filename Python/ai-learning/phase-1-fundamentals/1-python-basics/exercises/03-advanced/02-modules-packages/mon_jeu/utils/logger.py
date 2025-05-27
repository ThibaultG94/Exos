"""
Module de logging
"""
from datetime import datetime
import os

def log_evenement(message):
    log_path = os.path.join(os.path.dirname(__file__), 'game.log')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")