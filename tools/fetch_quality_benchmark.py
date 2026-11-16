"""Download pinned FER2013 mirror files with SHA-256 verification; no uploads."""
import argparse,hashlib,json,os
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,default=Path(os.environ.get('EMO_BENCHMARK_DIR',ROOT/'datasets/fer2013_benchmark')))
p.add_argument('--include-train',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
source=json.loads((ROOT/'reports/reliability/benchmark_sources.json').read_text())
for item in source['files']:
 name=item['file']
 if Path(name).name!=name:raise ValueError('Invalid filename')
 if name.startswith('train-') and not a.include_train:continue
 target=a.out/name
 if target.exists():
  if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Existing file checksum mismatch: '+name)
  print('Verified',name);continue
 temporary=target.with_suffix(target.suffix+'.part')
 with urllib.request.urlopen(item['url'],timeout=60) as r:temporary.write_bytes(r.read())
 if hashlib.sha256(temporary.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Download checksum mismatch: '+name)
 temporary.replace(target);print('Downloaded and verified',name)
(a.out/'benchmark_sources.json').write_text(json.dumps(source,indent=2))
