#!/bin/bash
# Entrypoint script voor QBN GPU container
# Test database connectie en start container

set -e

echo ""
echo "=========================================="
echo "   QBN (QuantBayes Nexus) Container"
echo "=========================================="
echo ""
echo "💡 EXEC COMMANDO (in Docker Desktop Exec tab):"
echo "   python docker-menu.py"
echo ""
echo "=========================================="

# Test database connectie
echo "🔧 Test database connectie..."
python3 -c "
from database.db import get_cursor
try:
    with get_cursor() as cur:
        cur.execute('SELECT 1')
        print('✅ Database connectie OK')
except Exception as e:
    print(f'⚠️ Database connectie failed: {e}')
    print('   Container start wel door, maar database is niet bereikbaar')
"

# Test GPU beschikbaarheid
echo "🔧 Test GPU beschikbaarheid..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU beschikbaar: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ Geen GPU gevonden - CPU mode actief')
"

echo "=========================================="
echo "🚀 Container klaar voor gebruik"
echo "=========================================="

# Check of we de real-time loop automatisch moeten starten
if [ "$START_INFERENCE_LOOP" = "true" ]; then
    echo "🌀 Starten van real-time inference loop..."
    python3 scripts/run_inference_loop.py &
fi

# Als geen argumenten: houd container draaiende
if [ $# -eq 0 ]; then
echo ""
echo "📌 Container draait in achtergrond."
echo "   Open menu via Docker Desktop Exec tab:"
echo "   python docker-menu.py"
echo ""
    # Houd container draaiende met tail (voor background mode)
    exec tail -f /dev/null
else
    exec "$@"
fi

