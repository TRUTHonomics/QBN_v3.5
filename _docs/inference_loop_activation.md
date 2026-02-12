# Inference Loop Monitoring & Activation

## Probleem
Slechts 18 entry predictions in afgelopen 30 dagen (0.6 per dag). Dit is te weinig voor:
- Prediction accuracy validatie (vereist min. 50 samples)
- Realtime trade decision making
- Model performance monitoring

## Doel
Inference loop moet continu draaien en predictions schrijven naar `qbn.output_entry` bij elke relevante signal update.

## Implementatie Status

### Inference Loop Locatie
- Service: `services/inference_loop.py`
- Trigger: PostgreSQL NOTIFY op `rolling_signal_update` channel
- Output: `qbn.output_entry` (entry predictions) en `qbn.output_position` (position updates)

### Check Inference Loop Status

```bash
# 1. Check of container draait
docker ps | grep QBN_v4

# 2. Check of inference_loop proces actief is
docker exec QBN_v4_Dagster_Webserver ps aux | grep inference_loop

# 3. Check recente predictions
docker exec QBN_v4_Dagster_Webserver python -c "
from database.db import get_cursor
with get_cursor() as cur:
    cur.execute('''
        SELECT count(*), max(time) as latest
        FROM qbn.output_entry
        WHERE asset_id = 1 AND time >= now() - interval \'7 days\'
    ''')
    row = cur.fetchone()
    print(f'Predictions last 7 days: {row[0]}')
    print(f'Latest: {row[1]}')
"

# 4. Check inference loop logs
docker exec QBN_v4_Dagster_Webserver ls -lth /app/_log/*inference_loop* | head -5
docker exec QBN_v4_Dagster_Webserver tail -50 /app/_log/$(ls -t /app/_log/*inference_loop*.log | head -1)
```

### Starten Inference Loop (als niet actief)

**Methode 1: Via Container Entrypoint**
```bash
# Check container startup command
docker inspect QBN_v4_Dagster_Webserver | grep -A 10 Cmd

# Herstart container (als inference_loop in entrypoint zit)
docker restart QBN_v4_Dagster_Webserver
```

**Methode 2: Handmatig Starten**
```bash
# Start inference loop in background
docker exec -d QBN_v4_Dagster_Webserver python -m services.inference_loop --asset-id 1

# Of via nohup voor persistentie
docker exec -d QBN_v4_Dagster_Webserver nohup python -m services.inference_loop --asset-id 1 > /app/_log/inference_loop_manual.log 2>&1 &
```

**Methode 3: Via Dagster Job (als gedefinieerd)**
```bash
# Check Dagster sensor/schedule voor inference
docker exec QBN_v4_Dagster_Webserver dagster sensor list

# Start sensor als beschikbaar
docker exec QBN_v4_Dagster_Webserver dagster sensor start qbn_inference_sensor
```

### Verificatie

Na starten, wacht 5-10 minuten en check:

```bash
# Check nieuwe predictions
docker exec QBN_v4_Dagster_Webserver python -c "
from database.db import get_cursor
from datetime import datetime, timedelta
with get_cursor() as cur:
    cur.execute('''
        SELECT time, trade_hypothesis, prediction_1h, prediction_4h, prediction_1d
        FROM qbn.output_entry
        WHERE asset_id = 1 AND time >= %s
        ORDER BY time DESC
        LIMIT 10
    ''', (datetime.now() - timedelta(hours=1),))
    rows = cur.fetchall()
    print(f'Predictions last hour: {len(rows)}')
    for r in rows:
        print(f'  {r[0]}: TH={r[1]}, 1h={r[2]}, 4h={r[3]}, 1d={r[4]}')
"
```

### Troubleshooting

**Geen predictions maar loop draait wel:**
1. Check of `rolling_signals_current` updates krijgt:
   ```sql
   SELECT asset_id, rolling_60m_start, updated_at
   FROM kfl.rolling_signals_current
   WHERE asset_id = 1;
   ```

2. Check of NOTIFY trigger werkt:
   ```sql
   -- Handmatig NOTIFY versturen voor test
   NOTIFY rolling_signal_update, '{"asset_id": 1}';
   ```

3. Check inference loop subscriptie:
   ```python
   # In inference_loop.py moet LISTEN query aanwezig zijn:
   # cur.execute("LISTEN rolling_signal_update")
   ```

**CPT cache leeg:**
Als inference loop CPTs niet kan laden:
```bash
# Genereer CPTs
docker exec QBN_v4_Dagster_Webserver python -m scripts.qbn_pipeline_runner --asset-id 1
```

### Monitoring Metrics

Verwachte frequency:
- **1-minute candles**: ~1440 candles/dag = potentieel 1440 inference calls
- **Signal updates**: Alleen bij significante veranderingen → ~50-200 updates/dag verwacht
- **Entry predictions**: Bij elke update → 50-200 predictions/dag verwacht

Als <10 predictions/dag: inference loop draait niet of ontvangt geen triggers.

## Productie Configuratie

Voor 24/7 operatie:

1. **Container restart policy**:
   ```yaml
   # docker-compose.yml
   services:
     qbn_v4_dagster_webserver:
       restart: unless-stopped
   ```

2. **Health check**:
   ```yaml
   healthcheck:
     test: ["CMD", "python", "-c", "from database.db import get_cursor; cur=get_cursor().__enter__(); cur.execute('SELECT 1')"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```

3. **Inference loop als systemd service** (alternatief):
   ```ini
   [Unit]
   Description=QBN Inference Loop
   After=docker.service
   
   [Service]
   Type=simple
   ExecStart=/usr/bin/docker exec QBN_v4_Dagster_Webserver python -m services.inference_loop --asset-id 1
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```

## Conclusie

**Acties:**
1. ✅ Check inference loop status (zie commands hierboven)
2. ✅ Start loop als niet actief (methode 2 of 3)
3. ✅ Verificeer predictions na 10 min
4. ✅ Als >50 predictions/dag: probleem opgelost
5. ⚠️  Als <10 predictions/dag: check NOTIFY trigger en rolling_signals updates

**Verwacht resultaat:**
- 50-200 entry predictions per dag (vs huidige 0.6/dag)
- Prediction accuracy validatie wordt mogelijk (vereist ≥50 samples)
- Real-time trade signaling actief

**Timestamp:** 2026-02-08 09:30
