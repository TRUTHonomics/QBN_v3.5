#!/usr/bin/env python3
"""
Analyseer weight verschillen per horizon in qbn.signal_weights.

Doel: Bepaal of horizon-specifieke weight lookup implementatie de moeite waard is.
"""

import psycopg2

def main():
    conn = psycopg2.connect('host=10.10.10.3 port=5432 dbname=kflhyper user=cursor_ai password=1234')
    cur = conn.cursor()

    # Haal weight verschillen per signaal op
    query = """
    SELECT 
        signal_name,
        MAX(CASE WHEN horizon = '1h' THEN weight END) as w_1h,
        MAX(CASE WHEN horizon = '4h' THEN weight END) as w_4h,
        MAX(CASE WHEN horizon = '1d' THEN weight END) as w_1d,
        MAX(CASE WHEN horizon = '1h' THEN weight END) - MAX(CASE WHEN horizon = '4h' THEN weight END) as delta_1h_4h,
        MAX(CASE WHEN horizon = '1h' THEN weight END) - MAX(CASE WHEN horizon = '1d' THEN weight END) as delta_1h_1d
    FROM qbn.signal_weights
    WHERE asset_id = 1 
      AND run_id = (SELECT run_id FROM qbn.signal_weights WHERE asset_id = 1 ORDER BY last_trained_at DESC LIMIT 1)
    GROUP BY signal_name
    HAVING 
        MAX(CASE WHEN horizon = '1h' THEN weight END) IS NOT NULL
    ORDER BY ABS(MAX(CASE WHEN horizon = '1h' THEN weight END) - COALESCE(MAX(CASE WHEN horizon = '4h' THEN weight END), 0)) DESC NULLS LAST
    LIMIT 30;
    """

    cur.execute(query)
    rows = cur.fetchall()

    print('Top 30 signalen met grootste weight verschillen per horizon:')
    print('=' * 90)
    print(f"{'Signal':<30} {'1h':>8} {'4h':>8} {'1d':>8} {'Δ(1h-4h)':>10} {'Δ(1h-1d)':>10}")
    print('-' * 90)

    significant_diffs = 0
    max_delta = 0.0
    
    for row in rows:
        sig, w1h, w4h, w1d, d_4h, d_1d = row
        w1h = w1h or 0
        w4h = w4h or 0
        w1d = w1d or 0
        d_4h = d_4h or 0
        d_1d = d_1d or 0
        
        max_delta = max(max_delta, abs(d_4h), abs(d_1d))
        
        if abs(d_4h) > 0.1 or abs(d_1d) > 0.1:
            significant_diffs += 1
        
        print(f'{sig:<30} {w1h:>8.3f} {w4h:>8.3f} {w1d:>8.3f} {d_4h:>10.3f} {d_1d:>10.3f}')

    print('-' * 90)
    print(f'Signalen met significant verschil (>0.1): {significant_diffs}/{len(rows)}')
    print(f'Maximum absolute delta: {max_delta:.3f}')

    # Count totaal aantal signalen
    cur.execute('SELECT COUNT(DISTINCT signal_name) FROM qbn.signal_weights WHERE asset_id = 1')
    total_signals = cur.fetchone()[0]
    print(f'Totaal signalen in database: {total_signals}')

    if significant_diffs > 0:
        pct = (significant_diffs / total_signals) * 100
        print(f'Percentage met significant verschil: {pct:.1f}%')
        
        print('\n' + '=' * 90)
        if pct > 20:
            print('✅ AANBEVELING: Implementeer horizon-specifieke weights (>20% verschil)')
            print('   Impact: HOOG - significante verbetering verwacht voor 4h/1d predictions')
        elif pct > 5:
            print('⚠️  AANBEVELING: Overweeg implementatie (5-20% verschil)')
            print('   Impact: MEDIUM - matige verbetering mogelijk')
        else:
            print('❌ AANBEVELING: Skip implementatie (<5% verschil, low ROI)')
            print('   Impact: LOW - verwaarloosbare verbetering')
    else:
        print('\n❌ Geen significante verschillen gevonden - implementatie niet nodig')

    conn.close()

if __name__ == '__main__':
    main()
