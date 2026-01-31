"""Centrale utility voor configuratie-gerelateerde warnings."""
import logging
from typing import Dict, Any

_warned_configs: set = set()  # Voorkom spam

def warn_fallback_active(
    component: str,
    config_name: str,
    fallback_values: Dict[str, Any],
    reason: str,
    fix_command: str = None
):
    """Log WARNING wanneer fallback config actief is."""
    cache_key = f"{component}:{config_name}"
    if cache_key in _warned_configs:
        return  # Al gelogd deze sessie
    _warned_configs.add(cache_key)
    
    logger = logging.getLogger(component)
    fix_hint = f"\n    FIX: {fix_command}" if fix_command else ""
    
    logger.warning(
        f"\n{'⚠️'*20}\n"
        f"FALLBACK CONFIGURATIE ACTIEF\n"
        f"{'⚠️'*20}\n"
        f"Component:  {component}\n"
        f"Config:     {config_name}\n"
        f"Reden:      {reason}\n"
        f"Waarden:    {fallback_values}"
        f"{fix_hint}\n"
        f"{'⚠️'*20}\n"
    )
