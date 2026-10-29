"""Coverage-first figures and mask diagnostics for a frozen RP stress baseline."""
import argparse,collections,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    report=json.loads((out/'report.json').read_text());p=json.loads((out/'protocol.json').read_text());parent=Path(p['temporal_mask_extension']['parent'])
    original_denominators=collections.defaultdict(collections.Counter)
    with gzip.open(str(out/'annotations.jsonl.gz'),'rt') as source:
        for r in map(json.loads,source):
            for s in r['sites']:
                for dimension,key in [('all','all'),('item',r['item']),('phone',s['expected'])]:
                    c=original_denominators[dimension,key]
                    if s['status']=='normal':c['normal']+=1
                    elif s['status'] in ('substitution','deletion'):c['error']+=1
    outer=[]
    for row in report['site_results']:
        c=row['counts'];den=original_denominators[row['dimension'],row['key']];n,e=den['normal'],den['error']
        assert n>=c.get('normal',0) and e>=c.get('error',0)
        outer.append(dict(band_samples=row['band_samples'],policy=row['policy'],dimension=row['dimension'],key=row['key'],
            original_normal=n,original_error=e,normal_coverage=c.get('observed_normal',0)/n if n else None,
            error_coverage=c.get('observed_error',0)/e if e else None,
            fpr_lower=c.get('fp',0)/n if n else None,fpr_upper=(c.get('fp',0)+n-c.get('observed_normal',0))/n if n else None,
            exact_lower=c.get('exact',0)/e if e else None,exact_upper=(c.get('exact',0)+e-c.get('observed_error',0))/e if e else None))
    (out/'original_denominator_bounds.json').write_text(json.dumps(dict(
        scope='All originally identified normal/errors from section 22, including annotation-boundary exclusions; unknown truth remains unknown',
        note='Marginal conservative bounds, not confidence intervals or jointly attainable alignment endpoints',rows=outer),ensure_ascii=False,indent=2)+'\n')
    a1=gzip.open(str(parent/'evaluated.jsonl.gz'),'rt');a2=gzip.open(str(out/'evaluated.jsonl.gz'),'rt');reasons=collections.defaultdict(collections.Counter);n=0
    for l1,l2 in zip(a1,a2):
        r1,r2=json.loads(l1),json.loads(l2);assert r1['uid']==r2['uid'] and r1['predicted']==r2['predicted'];n+=1
        for s1,s in zip(r1['sites'],r2['sites']):
            assert s1['annotation_exclusion']==s['annotation_exclusion']
            if s['annotation_exclusion']:reason='annotation_'+s['annotation_exclusion']
            elif s['source_unknown_time_mask']:reason='unknown_time_overlap'
            elif s['score'] is None:reason='no_ctc_score'
            elif len(s['prediction_options'])!=1:reason='prediction_correspondence_ambiguous'
            elif s['prediction_options'][0]=='<DEL>' and not s['prediction_deletion_anchored']:reason='prediction_deletion_unanchored'
            elif s['nearest_seam_samples']<=640:reason='chunk_boundary_mask'
            else:reason='observed'
            for item in ('all',r2['item']):reasons[item][reason]+=1
    a1.close();a2.close();assert n==report['recordings']==1118
    (out/'mask_diagnostics.json').write_text(json.dumps(dict(records_with_identical_predictions=n,exclusive_mask_reasons=dict(reasons),
        note='scoring masks change coverage only; neither version changes model predictions'),ensure_ascii=False,indent=2)+'\n')
    rows={r['key']:r for r in report['site_results'] if r['band_samples']==640 and r['policy']=='seen_CAL_fixed' and r['dimension']=='item'}
    plot_bounds={r['key']:r for r in outer if r['band_samples']==640 and r['policy']=='seen_CAL_fixed' and r['dimension']=='item'}
    items=sorted(rows);y=np.arange(len(items));fig,axes=plt.subplots(1,2,figsize=(11,5))
    for j,(key,label) in enumerate([('annotation_coverage','Boundary-eligible / known reference'),('normal_coverage','Observed / original identified normal'),('error_coverage','Observed / original identified error')]):
        values=[100*(rows[k]['rates'][key] if j==0 else plot_bounds[k][key]) for k in items]
        axes[0].barh(y+(j-1)*.22,values,height=.2,label=label)
    axes[0].set_yticks(y);axes[0].set_yticklabels(items);axes[0].set_xlim(0,100);axes[0].set_xlabel('Coverage (%)');axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.2))
    for j,item in enumerate(items):
        bound=next(x for x in outer if x['key']==item and x['dimension']=='item' and x['policy']=='seen_CAL_fixed' and x['band_samples']==640)
        rates=rows[item]['rates'];low=100*bound['fpr_lower'];high=100*bound['fpr_upper'];point=100*rates['observed_normal_fpr']
        axes[1].plot([low,high],[j,j],color='gray',lw=5,alpha=.5);axes[1].plot(point,j,'o',color='tab:red')
        axes[1].annotate('%.2f%% observed'%point,(point,j),xytext=(0,10),textcoords='offset points',fontsize=8)
    axes[1].set_yticks(y);axes[1].set_yticklabels(items);axes[1].set_xlabel('False-positive rate (%)');axes[1].axvline(1,color='black',ls='--',lw=1)
    axes[1].set_ylim(-.6,len(items)-.4);axes[1].set_title('Gray: original-denominator bounds, not CI',fontsize=10)
    fig.suptitle('Frozen B0 / original seen CAL threshold / RP secondary stress');fig.tight_layout(rect=(0,.12,1,.98))
    fig.savefig(out/'coverage_and_fpr.svg');fig.savefig(out/'coverage_and_fpr.png',dpi=140);plt.close(fig)
    for item in items:
        r=rows[item];print(item,json.dumps(dict(counts=r['counts'],rates=r['rates']),ensure_ascii=False))

if __name__=='__main__':main()
