"""Recreate the full noisy sample and compare data/weights to the final flagged MS."""
import json
import shutil
from pathlib import Path
import numpy as np
from casatools import table
from scripts.create_dataset_v1 import (THERMAL_DIR, NoiseControl, _apply_noise_control,
    _noise_request, _stable_seed, _thermal_source)
from scripts.simulation import add_thermal_noise_inplace, initialize_weights

run=Path('runs/uv-angular-2241-noise50-final')
source_id='2241+098'
manifest=json.loads((THERMAL_DIR/'report.json').read_text())
source=_thermal_source(THERMAL_DIR,manifest,source_id)
model,params,sigma=_noise_request(source)
v_ms=run/'work'/source_id/f'{source_id}_V.ms'
final=run/f'{source_id}.ms'
audit=run/'audit_unmasked.ms'
if audit.exists():raise FileExistsError(audit)
shutil.copytree(v_ms,audit)
try:
 seed=_stable_seed(source_id,'thermal_noise')
 control=NoiseControl(50.,'noise_snr_50')
 injected=_apply_noise_control(source,control,v_ms,audit,sigma,model,params,seed)
 add_thermal_noise_inplace(audit,noise_model=model,noise_parameters=params,seed=seed)
 initialize_weights(audit,injected['final_total_sigma_jy'])
 left,right=table(),table();left.open(str(audit),nomodify=True);right.open(str(final),nomodify=True)
 try:
  common=set(left.colnames())&set(right.colnames())
  columns=[c for c in ('DATA','CORRECTED_DATA','WEIGHT','SIGMA','WEIGHT_SPECTRUM','SIGMA_SPECTRUM') if c in common]
  for c in columns:
   for start in range(0,left.nrows(),2048):
    n=min(2048,left.nrows()-start)
    np.testing.assert_array_equal(left.getcol(c,startrow=start,nrow=n),right.getcol(c,startrow=start,nrow=n),err_msg=f'{c} rows {start}:{start+n}')
 finally:left.close();right.close()
 result=json.loads((run/'validation.json').read_text())
 result['noise_added_before_uv_flags']=True
 result['unmasked_reconstruction_matches_final_data_and_weights']=True
 result['identical_columns']=columns
 result['reconstructed_extra_snr_corr']=injected['SNR_corr_measured']
 (run/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Full unmasked reconstruction matches masked MS:',columns,flush=True)
finally:
 if audit.exists():shutil.rmtree(audit)
