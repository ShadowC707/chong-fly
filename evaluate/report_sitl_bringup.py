"""Summarize all declared physics attempts, including aborted runs."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('output/sitl_bringup_v1'))
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
    rows=[]
    for series in ('physics_01','physics_02'):
        for i,case in enumerate(('hover','roll','pitch','yaw')):
            folder=args.root/series/case
            report=json.loads((folder/'report.json').read_text())
            rows.append({'series':series,**report})
            trace=np.load(folder/'trace.npz')
            if len(trace['time_s']):
                t,s=trace['time_s'],trace['state']
                axis={'hover':8,'roll':6,'pitch':7,'yaw':8}[case]
                label=series+(' completed' if report['completed'] else ' ABORTED')
                axes[i,0].plot(t,s[:,2],label=label)
                axes[i,1].plot(t,np.degrees(s[:,axis]),label=label)
            else:
                axes[i,0].plot([],[],label=series+' FAILED BEFORE RELEASE')
            axes[i,0].set_ylabel(case)
    for i in range(4):
        axes[i,0].set_title('Height (m), no altitude hold')
        axes[i,1].set_title('Measured axis angle (deg)')
        for axis in axes[i]:
            axis.set_xlabel('Physics time (s)')
            axis.set_xlim(0,4)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
    fig.suptitle('Actual Betaflight SITL + PyBullet; experimental, timing NOT validated')
    fig.savefig(args.root/'comparison.png',dpi=140)
    plt.close(fig)
    summary={'scope':'bring-up only; NOT a neural policy or flight validation',
        'runs':len(rows),'completed':sum(r['completed'] for r in rows),
        'integration_admitted':False,'results':rows}
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))


if __name__=='__main__': main()
