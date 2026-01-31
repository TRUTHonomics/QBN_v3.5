"""
Watchdog Service voor QBN Inference Loop.
Bewaakt de hartslag van de InferenceLoopService en herstart deze indien nodig.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

class InferenceLoopWatchdog:
    """
    Watchdog die controleert of de inference loop nog leeft.
    """
    
    def __init__(self, service, timeout_seconds: int = 60):
        self.service = service
        self.timeout_seconds = timeout_seconds
        self.running = False
        self._watch_task = None

    async def start(self):
        """Start de watchdog monitoring."""
        logger.info(f"🛡️ Watchdog gestart (Timeout: {self.timeout_seconds}s)")
        self.running = True
        self._watch_task = asyncio.create_task(self._watch())

    async def stop(self):
        """Stop de watchdog."""
        self.running = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    async def _watch(self):
        """Monitor de hartslag van de service."""
        while self.running:
            await asyncio.sleep(10) # Check elke 10 seconden
            
            last_hb = self.service.get_heartbeat()
            if not last_hb:
                continue
                
            elapsed = (datetime.now(timezone.utc) - last_hb).total_seconds()
            
            if elapsed > self.timeout_seconds:
                logger.error(f"🚨 WATCHDOG ALERT: Inference loop lijkt vast te lopen! ({elapsed:.1f}s sinds laatste hartslag)")
                await self._recover()

    async def _recover(self):
        """Triggers recovery van de service."""
        logger.warning("Attempting recovery van de Inference Loop...")
        try:
            # We herstarten de listener component van de service
            # (In een echte productieomgeving zou je hier het hele proces kunnen killen 
            # zodat Docker/systemd het herstart, maar we proberen het hier intern)
            await self.service.stop()
            await asyncio.sleep(2)
            asyncio.create_task(self.service.start())
            logger.info("✅ Recovery gestart")
        except Exception as e:
            logger.error(f"❌ Recovery gefaald: {e}")

