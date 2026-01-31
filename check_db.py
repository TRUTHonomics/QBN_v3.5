from database.db import get_cursor
with get_cursor() as cur:
    cur.execute('SELECT COUNT(*) FROM qbn.signal_weights')
    print(f'Weights: {cur.fetchone()[0]}')
    cur.execute('SELECT COUNT(*) FROM qbn.signal_classification')
    print(f'Classification: {cur.fetchone()[0]}')
    cur.execute('SELECT COUNT(*) FROM qbn.signal_outcomes')
    print(f'Outcomes: {cur.fetchone()[0]}')

