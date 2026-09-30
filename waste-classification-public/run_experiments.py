"""After download, prepare and audit review: python run_experiments.py"""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--out',default='outputs')
p.add_argument('--batch-size',type=int,default=32)
a=p.parse_args()
script=str(Path(__file__).with_name('research.py'))
for command,extra in [('train',['--batch-size',str(a.batch_size)]),('benchmark',[]),('report',[])]:
    subprocess.run([sys.executable,script,command,'--out',a.out,*extra],check=True)
print('Complete. Open',Path(a.out).resolve()/'paper_ready'/'results.html')
