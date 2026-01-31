# Runbook: Fase 1 QBN_v3 Afronding

Dit document beschrijft de uitvoeringsvolgorde voor het afronden van Fase 1.

## Vereisten

- SSH tunnel naar database server (10.10.10.1:5432 -> localhost:15432)
- PostgreSQL client (psql) of MCP database tools
- QBN_v3 container draait (poort 8082)

## Uitvoeringsvolgorde

### Stap 1: Deploy trigger update

De triggers in `008_mtf_triggers_complete.sql` zijn aangepast om `atr_14` uit `indicators_unified_cache` te halen.

```bash
# Op de database server of via psql:
psql -h 10.10.10.1 -U postgres -d kflhyper -f S:/kfl_backend_v3/src/database/triggers/008_mtf_triggers_complete.sql
```

**Verificatie:** Nieuwe records in MTF tabellen moeten nu automatisch `atr_at_signal` gevuld krijgen.

### Stap 2: Signal Configuration Sync (Aangepast per 2026-01-06)

**NIEUW:** De YAML configuraties staan nu lokaal in de QBN container (`/app/kfl_backend_config/`).
QBN_v3 is de **config master** - alle andere containers lezen uit de database.

1. **Database Migratie (eenmalig):**
   ```bash
   psql -h 10.10.10.1 -U postgres -d kflhyper -f f:/Containers/QBN_v3/database/migrations/013_create_signal_discretization.sql
   ```

2. **Config Sync (vanuit QBN_v3 container):**
   ```bash
   docker exec QBN_v3_Validation python scripts/db_sync.py
   ```
   
   **REASON:** Dit synct de lokale YAMLs naar `qbn.signal_classification` en `qbn.signal_discretization`.
   De sync wordt automatisch aangeroepen na threshold-analyse via `ConfigPersister`.

3. **Validatie:**
   ```bash
   docker exec QBN_v3_Validation python scripts/validate_signal_classification.py
   ```

4. **Config Locaties:**
   - Container: `/app/kfl_backend_config/`
   - Host: `F:/Containers/QBN_v3/kfl_backend_config/`

### Stap 3: ATR Backfill uitvoeren

Vul de NULL `atr_at_signal` waarden voor bestaande ~87M records.

```bash
psql -h 10.10.10.1 -U postgres -d kflhyper -f f:/Containers/QBN_v3/scripts/001_backfill_atr_at_signal.sql
```

**WAARSCHUWING:** Dit kan lang duren (uren). Overweeg de batch versie in het script te gebruiken.

### Stap 4: Outcome constraints toevoegen

```bash
psql -h 10.10.10.1 -U postgres -d kflhyper -f f:/Containers/QBN_v3/scripts/002_add_outcome_constraints.sql
```

### Stap 5: Performance indexen toevoegen

```bash
psql -h 10.10.10.1 -U postgres -d kflhyper -f f:/Containers/QBN_v3/scripts/003_add_outcome_indexes.sql
```

### Stap 6: Outcome backfill testen

In de QBN_v3 container:

```bash
docker exec -it QBN_v3 python scripts/outcome_backfill.py --status
docker exec -it QBN_v3 python scripts/outcome_backfill.py --asset BTCUSDT --horizon 1h --dry-run
```

### Stap 7: Validation scripts testen

```bash
# Outcome backfill validation
docker exec -it QBN_v3 python scripts/validate_outcome_backfill.py --all

# Signal classification & discretization sync validation  
docker exec -it QBN_v3 python scripts/validate_signal_classification.py
```

## Verwachte output

### validate_signal_classification.py (Sync Validation)

```
✅ Check 1: Database Sync Integrity (YAML ↔ qbn schema)
   ✅ qbn.signal_classification matcht met YAML (125 signalen)
   ✅ qbn.signal_discretization matcht volledig met YAML
✅ Check 2: MTF Column Mapping (YAML ↔ kfl schema)
   ✅ LEADING: Alle YAML signalen aanwezig in kfl.mtf_signals_lead
   ...
✅ ALL CHECKS PASSED
```

## Traceability & Reproduceerbaarheid

Sinds de update van 2026-01-04 wordt bij elke CPT training run in QBN_v3 automatisch een snapshot van de actieve database configuratie gemaakt:
- **Pad:** `_validation/config_snapshots/config_snapshot_asset_[ID]_[TS].json`
- **Inhoud:** Volledige dump van `qbn.signal_classification` en `qbn.signal_discretization`.

Dit garandeert dat we altijd kunnen achterhalen met welke drempelwaarden een specifiek model is getraind.
