from database.db import get_cursor
with get_cursor() as cur:
    cur.execute('SELECT indicator_base, threshold_name, threshold_value FROM qbn.signal_discretization')
    rows = cur.fetchall()
    for r in rows:
        print(f"{r[0]}: {r[1]} = {r[2]}")

