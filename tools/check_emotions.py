"""Run the actual service analyzer on one dataset frame, without queues or S3."""
import argparse
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for folder in ['ai-service/src', 'libs/common/src']:
    sys.path.insert(0, str(ROOT / folder))
from common.messages import BoundingBox, Face
from ai_service.analyzer import analyze_emotions

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, default=ROOT / 'datasets/faces')
p.add_argument('--id', help='Optional dataset row ID; defaults to the first row')
a = p.parse_args()
with (a.data / 'dataset.csv').open(encoding='utf-8', newline='') as f:
    row = next((r for r in csv.DictReader(f) if a.id is None or r['id'] == a.id), None)
if row is None:
    raise SystemExit('Dataset is empty or requested ID does not exist')
box = BoundingBox(**{k: int(row[k]) for k in ['x', 'y', 'width', 'height']})
print('Frame:', row['file'], flush=True)
print('Video label (not verified frame truth):', row['emotion'], flush=True)
print(analyze_emotions((a.data / row['file']).read_bytes(), [Face(bbox=box)]))
