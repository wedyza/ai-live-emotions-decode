"""Fetch immutable-by-checksum FER+ inputs; no uploads."""
import os,json,hashlib,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EMO_CONSENSUS_DIR',ROOT/'datasets/ferplus_consensus')).resolve();P.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'tools/fetch_quality_benchmark.py'),'--include-train'],check=True)
source=json.loads((ROOT/'reports/consensus/sources.json').read_text())
for row in source['files']:
 target=P/row['file']
 if target.exists():
  if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Existing source checksum mismatch')
  print('Verified',target.name);continue
 temporary=target.with_suffix(target.suffix+'.part')
 with urllib.request.urlopen(row['url'],timeout=60) as r,temporary.open('wb') as out:
  while True:
   chunk=r.read(1024*1024)
   if not chunk:break
   out.write(chunk)
 if hashlib.sha256(temporary.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Downloaded source checksum mismatch')
 temporary.replace(target);print('Ready',target.name)
(P/'release_gate.json').write_text((ROOT/'reports/consensus/release_gate.json').read_text())
