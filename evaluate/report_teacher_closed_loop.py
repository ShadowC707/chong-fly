"""Report current teacher flights against saved neural and historical teacher runs."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    out=Path(out)
    results=json.loads((out/'report.json').read_text())
    old=json.loads((ROOT/'output/maneuver_teacher_check_v2/report.json').read_text())
    prior=ROOT/'output/practical_maneuver_v1'
    neural=json.loads((prior/'report.json').read_text())
    assert len(results)==4
    fig,axes=plt.subplots(2,2,figsize=(12,11),layout='constrained')
    timefig,timeaxes=plt.subplots(4,2,figsize=(14,12),layout='constrained')
    stats={};table=[]
    for index,(ax,r) in enumerate(zip(axes.flat,results)):
        case=r['case'];dest=out/case;m=r['metrics']
        g=json.loads((dest/'manifest.json').read_text())['geometry']
        for b in g['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
        for c in g['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.85'))
        variants=[]
        for name,color in [('before','#2764b5'),('candidate','#d45123')]:
            rr=next(v for v in neural if v['case']==case and v['driver']==name)
            variants.append((name,prior/(case+'_'+name),rr['metrics'],color))
        variants.append(('teacher 1530',dest,m,'#339258'))
        for name,path,metrics,color in variants:
            with np.load(path/'physical.npz') as a:
                pos=a['pos'];ax.plot(*pos[:,:2].T,color=color,lw=1.2,label=f'{name}: {metrics["duration_s"]:.2f} s')
                ax.scatter(*pos[-1,:2],c=color,marker='x' if metrics['collision'] else 'o',s=35)
                ax.scatter(*pos[0,:2],c='black',marker='s',s=15)
        ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)',title=case)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
        with np.load(dest/'decisions.npz') as d,np.load(dest/'trace.npz') as tr:
            valid=d['auto'];teacher=d['teacher'];delta=teacher[:,3]-1500
            distance=d['X'][:,2:66].min(axis=1)*3.5
            n=int(valid.sum()); count=lambda mask:int((valid&mask).sum())
            stats[case]=dict(auto_decision_frames=n,
                zero_tof_drive_frames=count(distance>=.8),
                invisible_release_band_frames=count((distance>=.8)&(distance<.95)),
                held_in_invisible_band_frames=count((distance>=.8)&(distance<.95)&(d['phase']=='held_brake')),
                teacher_turn_left_frames=count(delta< -50),teacher_turn_right_frames=count(delta>50),
                teacher_phases={p:count(d['phase']==p) for p in np.unique(d['phase'])},
                observers={name:dict(pitch_mae_pwm=float(np.abs(d[name][valid,2]-teacher[valid,2]).mean()) if n else None,
                    yaw_mae_pwm=float(np.abs(d[name][valid,3]-teacher[valid,3]).mean()) if n else None) for name in ('before','candidate')})
            # Verify exported stage log is populated, not silently using wrong keys.
            accepted=np.isfinite(tr['accepted']).all(axis=1)
            stats[case]['accepted_command_samples']=int(accepted.sum())
            if n: assert accepted.any()
            for j,channel in enumerate((2,3)):
                axis=timeaxes[index,j]
                axis.plot(d['time'],teacher[:,channel],label='teacher raw',lw=1.)
                axis.plot(tr['time'],tr['applied'][:,channel],label='plant applied',lw=.9,alpha=.8)
                axis.axhline(1500,color='.5',ls=':',lw=.8)
                axis.set(title=case+(' | Pitch' if channel==2 else ' | Yaw'),xlabel='Time (s)',ylabel='PWM')
                axis.grid(alpha=.2);axis.legend(fontsize=8)
        table.append(dict(case=case,current=m,historical_teacher1600=old[case]['teacher']['metrics']))
    fig.suptitle('Current v5/1530 teacher vs saved neural flights | radius 0.20 m\n'
                 'Teacher drives from first call; native AUTO admission; X = contact')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    timefig.suptitle('Current teacher commands before and after the unchanged flight stack')
    timefig.savefig(out/'commands.png',dpi=150);plt.close(timefig)
    (out/'analysis.json').write_text(json.dumps(dict(comparison=table,behavior=stats),indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Teacher v5/1530: native AUTO admission','',
        '| Room | End (s) | Contact | AUTO (s) | Gap (cm) | AUTO speed (m/s) | Reverse/AUTO (%) |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in results:
        m=r['metrics'];speed=m['mean_auto_speed_mps']
        lines.append(f'| {r["case"]} | {m["duration_s"]:.3f} | {m["collision"] or "none"} | {m["auto_s"]:.3f} | '
            f'{100*m["min_clearance_m"]:.2f} | {speed if speed is not None else "no AUTO"} | {100*m["reverse_fraction_auto"]:.1f} |')
    lines+=['','Historical v5/1600 used baseline-assisted AUTO admission. The comparison does not isolate cruise Pitch alone.',
            'These four known starts are diagnostic, not flight qualification or proof of teacher completeness.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
