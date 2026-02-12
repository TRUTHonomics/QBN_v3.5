from __future__ import annotations

from dagster import AssetKey, AutoMaterializePolicy, MetadataValue, asset


# REASON: Alleen key (geen name) — Dagster: "Cannot specify name when key is provided"
# REASON: context zonder typehint — container-Dagster accepteert AssetExecutionContext niet
# REASON: postgres via context.resources — anders ziet Dagster "postgres" als input-asset
# REASON: required_resource_keys — materialization-job moet postgres resource krijgen
@asset(
    key=AssetKey(["phase1", "qbn_db_health_check"]),
    description="Fase 1 dummy asset: test Dagster ↔ Postgres connectie (geen GPU).",
    required_resource_keys={"postgres"},
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def qbn_db_health_check(context) -> dict:
    """
    Dummy asset die een simpele query doet.

    SSOT RULE: Test alleen met asset_id=9889.
    DB RULE: Gebruik schema-prefixen (qbn.).
    """
    postgres = context.resources.postgres

    query = """
    SELECT COUNT(*)::bigint AS cnt
    FROM qbn.barrier_outcomes
    WHERE asset_id = %s
    """

    with postgres.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (9889,))
            (cnt,) = cur.fetchone()

    context.log.info("qbn.barrier_outcomes count (asset_id=9889) = %s", cnt)
    context.add_output_metadata(
        {
            "asset_id": 9889,
            "row_count": MetadataValue.int(cnt),
        }
    )

    return {"asset_id": 9889, "row_count": int(cnt)}

