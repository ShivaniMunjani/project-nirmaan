import pandas as pd
import numpy as np
from datetime import datetime

def fetch_live_project_data():
    """
    This function SIMULATES fetching data from POWERGRID's live systems.
    """
    live_projects = [
        {'project_id': 'PG-2024-001', 'name': 'Mumbai High-Capacity Substation', 'lat': 19.07, 'lon': 72.87, 'status': 'Critical', 'progress': 45, 'predicted_delay_days': 65},
        {'project_id': 'PG-2023-045', 'name': 'Delhi-Kanpur Overhaul Line', 'lat': 28.70, 'lon': 77.10, 'status': 'On Track', 'progress': 78, 'predicted_delay_days': 5},
        {'project_id': 'PG-2024-015', 'name': 'Bangalore Smart Grid UG Cable', 'lat': 12.97, 'lon': 77.59, 'status': 'Critical', 'progress': 20, 'predicted_delay_days': 120},
        {'project_id': 'PG-2022-098', 'name': 'Kolkata East-West Corridor', 'lat': 22.57, 'lon': 88.36, 'status': 'At Risk', 'progress': 90, 'predicted_delay_days': 30},
        {'project_id': 'PG-2024-021', 'name': 'Chennai Industrial Power Hub', 'lat': 13.08, 'lon': 80.27, 'status': 'On Track', 'progress': 60, 'predicted_delay_days': 10}
    ]
    df_projects = pd.DataFrame(live_projects)
    df_projects['weather_alert'] = np.where(df_projects['name'].str.contains("Mumbai|Chennai"), "Heavy Rainfall Alert", "Clear")
    df_projects['vendor_status'] = np.random.choice(['Good', 'Delayed', 'Under Review'], size=len(df_projects), p=[0.6, 0.3, 0.1])
    return df_projects

def get_last_sync_time():
    """A simple function to show when the data was last "updated"."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")