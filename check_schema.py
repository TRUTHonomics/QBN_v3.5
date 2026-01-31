from database.db import get_cursor
with get_cursor() as cur:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'qbn' AND table_name = 'signal_discretization'")
    print([r[0] for r in cur.fetchall()])

