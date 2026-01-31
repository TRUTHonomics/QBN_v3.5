# time_close Migratie Instructies

## Overzicht

Deze migratie voegt `time_close` kolommen toe aan signals en MTF tabellen, en verwijdert de overbodige `time` kolom uit MTF tabellen.

## Benodigde Migraties

| Bestand | Beschrijving | Downtime Vereist |
|---------|--------------|------------------|
| `005_time_close_columns.sql` | Kolommen toevoegen + backfill | Nee |
| `006_time_close_pk_change.sql` | PK wijzigen + time verwijderen | **Ja** |
| `007_time_close_triggers.sql` | Trigger voor signals_current | Nee |
| `008_time_close_validation.sql` | Validatie + index updates | Nee |

## Uitvoervolgorde

### Stap 1: Kolommen toevoegen (kan tijdens productie)

```bash
psql -h 10.10.10.1 -U postgres -d KFLhyper -f 005_time_close_columns.sql
```

Dit voegt toe:
- `kfl.signals_current.time_close`
- `qbn.ml_multi_timeframe_signals.time_close_d/240/60/1`
- `qbn.ml_multi_timeframe_signals_cache.time_close_d/240/60/1`
- `staging.signals.time_close`
- `staging.mtf_signals.time_close_d/240/60/1`

### Stap 2: Triggers aanmaken (kan tijdens productie)

```bash
psql -h 10.10.10.1 -U postgres -d KFLhyper -f 007_time_close_triggers.sql
```

Dit maakt de `trg_signals_current_calculate_time_close` trigger aan.

### Stap 3: Python/Trigger code deployen

Deploy de bijgewerkte bestanden:
- `S:/kfl_backend_v2/src/database/triggers/005_mtf_direct_trigger.sql`
- `S:/kfl_backend_v2/src/pipeline/signals/models/signal_state.py`
- `f:/Containers/KFL_backend_GPU_v4/scripts/MTF/gpu_mtf_backfill.py`

### Stap 4: PK wijzigen (MAINTENANCE WINDOW)

**Stop alle applicaties die naar MTF tabellen schrijven!**

```bash
psql -h 10.10.10.1 -U postgres -d KFLhyper -f 006_time_close_pk_change.sql
```

Dit:
- Wijzigt PK van `(asset_id, time)` naar `(asset_id, time_1)`
- Verwijdert `time` kolom uit MTF tabellen
- Verwijdert oude indexes

### Stap 5: Validatie en cleanup

```bash
psql -h 10.10.10.1 -U postgres -d KFLhyper -f 008_time_close_validation.sql
```

## Gerelateerde Code Wijzigingen

### MTF Direct Trigger (`005_mtf_direct_trigger.sql`)
- `time` kolom verwijderd uit INSERT
- `time_close_d/240/60/1` toegevoegd

### GPU MTF Backfill (`gpu_mtf_backfill.py`)
- `build_mtf_signals()`: berekent `time_close_X` vectorized
- `save_to_staging()`: `time` verwijderd, `time_close_X` toegevoegd
- `merge_to_main()`: ON CONFLICT nu op `(asset_id, time_1)`
- `_get_existing_mtf_timestamps()`: gebruikt nu `time_1` ipv `time`

### SignalState Model (`signal_state.py`)
- `time_close: Optional[datetime] = None` toegevoegd

## Risico's

1. **Lock-tijd**: PK wijziging kan lang duren bij grote tabellen
2. **Queries**: Alle queries die `time` gebruiken zullen falen na migratie
3. **TimescaleDB**: Hypertable handling vereist mogelijk extra aandacht

## Rollback

Als er iets misgaat:

```sql
-- Alleen na stap 1 (kolommen toegevoegd, PK nog niet gewijzigd):
ALTER TABLE kfl.signals_current DROP COLUMN IF EXISTS time_close;
ALTER TABLE qbn.ml_multi_timeframe_signals DROP COLUMN IF EXISTS time_close_d;
-- etc.
```

Na de PK wijziging is rollback complexer en vereist database restore.

