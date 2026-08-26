import sqlite3
from pathlib import Path

import pandas as pd


def export_events(db_path: Path, out: Path) -> int:
    """events tablosunu Parquet/CSV'ye döker; satır sayısını döner."""
    db = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY id", db)
    finally:
        db.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    return len(df)
