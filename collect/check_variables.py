import pyvo

TAP_URL = "https://data-query.nrao.edu/tap"

svc = pyvo.dal.TAPService(TAP_URL)

# Find ObsCore table
tables = svc.run_sync("SELECT table_name FROM TAP_SCHEMA.tables").to_table().to_pandas()
obscore = [t for t in tables["table_name"] if "obscore" in t.lower()][0]

print("Using table:", obscore)

# List all columns
q = f"""
SELECT column_name, datatype, unit, ucd, description
FROM TAP_SCHEMA.columns
WHERE table_name = '{obscore}'
ORDER BY column_name
"""

cols = svc.run_sync(q).to_table().to_pandas()
print(cols)
