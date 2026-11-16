"""Require identical retained data and settings before comparing validation experiments."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('bbox',type=Path);p.add_argument('aligned',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
b=json.loads((a.bbox/'experiment.json').read_text());c=json.loads((a.aligned/'experiment.json').read_text())
for key in ['labels_sha256','train_ids','val_ids','seed','train_rows','val_rows']:
 if b[key]!=c[key]:raise ValueError('Runs not paired: '+key)
if b['smoke'] or c['smoke']:raise ValueError('Smoke runs are not quality comparisons')
result=dict(train_rows=b['train_rows'],val_rows=b['val_rows'],same_rows_and_subject_splits=True,test_evaluated=False,
 bbox_baseline_val=b['baseline_val'],aligned_baseline_val=c['baseline_val'],
 bbox_selected_val_macro_f1=b['best_val_macro_f1'],aligned_selected_val_macro_f1=c['best_val_macro_f1'],
 bbox_best_epoch=b['best_epoch'],aligned_best_epoch=c['best_epoch'],
 warning='Validation-only comparison with weak labels. Not evidence of improvement on new subjects. Epoch 0 means original weights retained.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
