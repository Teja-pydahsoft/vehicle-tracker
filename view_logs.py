import sqlite3
import pandas as pd
from tabulate import tabulate

def view_logs():
    try:
        conn = sqlite3.connect('gate_log.db')
        # Format the query to get the last 20 entries
        query = "SELECT timestamp, vehicle_type, track_id, direction, plate_number FROM vehicle_logs ORDER BY timestamp DESC LIMIT 20"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("\n[!] No logs found yet. Start the Vehicle Counter and process a video first!")
        else:
            print("\n--- LATEST GATE LOGS (LAST 20) ---")
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            print(f"\nTotal logs in database: {pd.read_sql_query('SELECT COUNT(*) FROM vehicle_logs', conn).iloc[0,0]}")
            
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    view_logs()
