import asyncio
import asyncpg
import json
import os

async def test_listen():
    db_config = {
        'host': '10.10.10.3',
        'port': 5432,
        'database': 'kflhyper',
        'user': 'pipeline',
        'password': 'pipeline123'
    }
    
    conn = await asyncpg.connect(**db_config)
    print("Connected to DB")
    
    def on_notify(connection, pid, channel, payload):
        print(f"Received on {channel}: {payload}")
        try:
            data = json.loads(payload)
            print(f"Parsed asset_id type: {type(data.get('asset_id'))}")
        except:
            print("Failed to parse JSON")

    await conn.add_listener('rolling_signal_update', on_notify)
    print("Listening on rolling_signal_update... (Waiting for 70s)")
    
    await asyncio.sleep(70)
    await conn.close()

if __name__ == "__main__":
    asyncio.run(test_listen())
