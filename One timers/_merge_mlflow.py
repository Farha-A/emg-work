"""Merge MLflow runs from source DB into destination DB.

Usage: python _merge_mlflow.py

Source:  Testing Different Approaches/mlflow.db
Dest:    Cleaning/mlflow.db
"""
import sqlite3
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlflow.db')
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlflow.db')

print(f"Source: {SRC}")
print(f"Dest:   {DST}")
print()

# Check both exist
if not os.path.exists(SRC):
    print(f"ERROR: Source DB not found: {SRC}")
    exit(1)
if not os.path.exists(DST):
    print(f"ERROR: Dest DB not found: {DST}")
    exit(1)

src_conn = sqlite3.connect(SRC)
dst_conn = sqlite3.connect(DST)

src_cur = src_conn.cursor()
dst_cur = dst_conn.cursor()

# Get all tables from source
src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
src_tables = [t[0] for t in src_cur.fetchall()]
print(f"Source tables: {src_tables}")

# Get all tables from dest
dst_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
dst_tables = [t[0] for t in dst_cur.fetchall()]
print(f"Dest tables:   {dst_tables}")
print()

# Tables to merge and their primary keys / unique constraints
# We use INSERT OR IGNORE to skip duplicates based on primary key
for table in src_tables:
    if table.startswith('alembic') or table.startswith('sqlite_'):
        print(f"Skipping {table}")
        continue

    # Get column info
    src_cur.execute(f"PRAGMA table_info([{table}])")
    columns = src_cur.fetchall()
    col_names = [c[1] for c in columns]

    # Count source rows
    src_cur.execute(f"SELECT COUNT(*) FROM [{table}]")
    src_count = src_cur.fetchone()[0]

    if src_count == 0:
        print(f"{table}: 0 rows in source, skipping")
        continue

    # Check if table exists in dest
    if table not in dst_tables:
        print(f"{table}: does NOT exist in dest, creating...")
        # Get the CREATE TABLE statement
        src_cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
        create_sql = src_cur.fetchone()[0]
        dst_cur.execute(create_sql)
        dst_conn.commit()

    # Get dest count before
    dst_cur.execute(f"SELECT COUNT(*) FROM [{table}]")
    dst_before = dst_cur.fetchone()[0]

    # Read all rows from source
    src_cur.execute(f"SELECT * FROM [{table}]")
    rows = src_cur.fetchall()

    # Insert with OR IGNORE to skip duplicates
    placeholders = ','.join(['?'] * len(col_names))
    col_list = ','.join([f'[{c}]' for c in col_names])
    insert_sql = f"INSERT OR IGNORE INTO [{table}] ({col_list}) VALUES ({placeholders})"

    for row in rows:
        try:
            dst_cur.execute(insert_sql, row)
        except sqlite3.IntegrityError:
            pass  # skip duplicates

    dst_conn.commit()

    # Get dest count after
    dst_cur.execute(f"SELECT COUNT(*) FROM [{table}]")
    dst_after = dst_cur.fetchone()[0]

    added = dst_after - dst_before
    print(f"{table}: {src_count} source rows, {dst_before} -> {dst_after} dest rows (+{added} new)")

src_conn.close()
dst_conn.close()

print("\nMerge complete!")
