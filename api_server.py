from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    """Serve the dashboard index file"""
    if os.path.exists("dashboard/index.html"):
        return FileResponse("dashboard/index.html")
    return {"status": "online", "message": "Smart Gate API is running, but dashboard/index.html was not found"}

# Mount the static files (CSS, JS, images)
if os.path.exists("dashboard"):
    app.mount("/static", StaticFiles(directory="dashboard"), name="static")

@app.get("/logs")
def get_logs(
    limit: int = 100, 
    vehicle_type: str = None,
    plate_number: str = None,
    start_date: str = None,
    end_date: str = None,
    start_time: str = "00:00:00",
    end_time: str = "23:59:59"
):
    """Get vehicle detection logs with optional filtering"""
    try:
        conn = get_db_connection()
        query = "SELECT * FROM vehicle_logs WHERE 1=1"
        params = []
        
        if vehicle_type and vehicle_type != "All":
            query += " AND vehicle_type LIKE ?"
            params.append(f"%{vehicle_type.lower()}%")
            
        if plate_number:
            query += " AND plate_number LIKE ?"
            params.append(f"%{plate_number.upper()}%")
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(f"{start_date} {start_time}")
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(f"{end_date} {end_time}")
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Replace NaN with empty string for cleaner JSON
        df = df.fillna("")
        
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

@app.get("/vehicle-types")
def get_vehicle_types():
    """Get unique vehicle types from logs"""
    try:
        conn = get_db_connection()
        query = "SELECT DISTINCT vehicle_type FROM vehicle_logs ORDER BY vehicle_type"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df["vehicle_type"].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
