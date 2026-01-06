from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI(title="Smart Gate API", description="API to access vehicle detection logs")

# Enable CORS (Allows your web dashboard to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = 'gate_log.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
def read_root():
    return {"status": "online", "message": "Smart Gate API is running"}

@app.get("/logs")
def get_logs(limit: int = 100, vehicle_type: str = None):
    """Get vehicle detection logs with optional filtering"""
    try:
        conn = get_db_connection()
        query = "SELECT * FROM vehicle_logs"
        params = []
        
        if vehicle_type:
            query += " WHERE vehicle_type = ?"
            params.append(vehicle_type)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get total counts for today"""
    try:
        conn = get_db_connection()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Total IN/OUT today
        query = """
            SELECT direction, COUNT(*) as count 
            FROM vehicle_logs 
            WHERE timestamp LIKE ? 
            GROUP BY direction
        """
        df = pd.read_sql_query(query, conn, params=(f"{today}%",))
        
        # Counts by vehicle type
        query_types = """
            SELECT vehicle_type, direction, COUNT(*) as count 
            FROM vehicle_logs 
            WHERE timestamp LIKE ? 
            GROUP BY vehicle_type, direction
        """
        df_types = pd.read_sql_query(query_types, conn, params=(f"{today}%",))
        
        conn.close()
        
        return {
            "date": today,
            "summary": df.to_dict(orient="records"),
            "by_type": df_types.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
