"""Audit nested angular flags and save initial UVW for the report (CASA)."""
import json
from pathlib import Path
import numpy as np
from casatools import table

run=Path('runs/uv-angular-2241-noise50-final')
source=run/'work'/'2241+098'/'2241+098_V.ms'
final=run/'2241+098.ms'
tb=table();tb.open(str(source),nomodify=True)
try:
 uvw=np.asarray(tb.getcol('UVW'),dtype=float)
 flags=np.asarray(tb.getcol('FLAG'),dtype=bool)
 row=np.asarray(tb.getcol('FLAG_ROW'),dtype=bool)
finally:tb.close()
usable=(~flags)&(~row.reshape(1,1,-1))
angle=np.mod(np.arctan2(uvw[1],uvw[0]),np.pi)
results=json.loads((run/'results.json').read_text())['rows']
for r in results:
 selected=angle<np.pi*r['angular_removed_percent']/100
 remaining=int((usable&~selected.reshape(1,1,-1)).sum())
 assert remaining==r['usable_remaining']
tb.open(str(final),nomodify=True)
try:
 actual_flags=np.asarray(tb.getcol('FLAG'),dtype=bool)
 actual_rows=np.asarray(tb.getcol('FLAG_ROW'),dtype=bool)
finally:tb.close()
expected=angle<np.pi*.9
np.testing.assert_array_equal(actual_flags,flags|expected.reshape(1,1,-1))
np.testing.assert_array_equal(actual_rows,row|expected)
rows_usable=np.any(~flags,axis=(0,1))&~row
extent=float(np.percentile(np.hypot(uvw[0][rows_usable],uvw[1][rows_usable]),99.5))*1.1
np.savez_compressed(run/'uvw_initial.npz',u=uvw[0],v=uvw[1],usable=rows_usable,extent=extent)
(run/'validation.json').write_text(json.dumps({'rows':len(results),'initial_usable_samples':int(usable.sum()),
 'final_usable_samples':int((usable&~expected.reshape(1,1,-1)).sum()),
 'all_remaining_counts_match':True,'final_flags_match_exact_90_degree_wedge':True,
 'nested_wedges':True,'fixed_image_grid':len({tuple(r['image_cell_arcsec']) for r in results})==1},indent=2)+'\n')
print('Audited',len(results),'cuts; final flags exact',flush=True)
