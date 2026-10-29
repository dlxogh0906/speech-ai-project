"""Report actual normalized context concentration independently of the fit objective."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
import argparse,collections,csv,json
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=a.out
    rows=list(csv.DictReader((p/'contexts.tsv').open(),delimiter='\t'));out=[]
    for arm in ('C0','C1','C2'):
        for subset in ('all','supported'):
            groups=collections.defaultdict(list)
            for row in rows:
                if row['arm']==arm and (subset=='all' or row['supported']=='True'):groups[row['phone'],row['position']].append(float(row['weighted_proxy']))
            values=[]
            for k,counts in groups.items():
                x=np.array(counts);prob=x/x.sum();hhi=float(sum(prob*prob))
                values.append(dict(phone=k[0],position=k[1],cells=len(x),HHI=hhi,excess_HHI=hhi-1/len(x),largest_share=float(max(prob))))
            out.append(dict(arm=arm,subset=subset,groups=len(groups),macro_excess_HHI=float(np.mean([v['excess_HHI'] for v in values])),
                macro_largest_share=float(np.mean([v['largest_share'] for v in values])),per_group=values))
    expected=p/'context_dispersion.json'
    if expected.exists():assert json.loads(expected.read_text())==out
    expected.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
