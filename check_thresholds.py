from database.db import get_cursor
with get_cursor() as cur:
    cur.execute('SELECT signal_name, low_threshold, high_threshold FROM qbn.signal_discretization')
    rows = cur.fetchall()
    print(f"{'Signal Name':<30} {'Low':>10} {'High':>10}")
    print("-" * 52)
    for r in rows:
        print(f"{r[0]:<30} {r[1]:>10} {r[2]:>10}")

