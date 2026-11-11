"""Compare saved native-RC, corrected-units and finer-step neural experiments."""
import argparse
import json
from pathlib import Path
import numpy as np

CASES=('clear','obstacle_left','obstacle_right','obstacle_center')


def run(root):
    out=root/'neural_review_v1';out.mkdir(exist_ok=True)
    sets=[('legacy','neural_units_v1','legacy'),('SITL native RC','neural_v1','sitl'),
          ('SITL units 4 ms','neural_units_v1','sitl'),('SITL units 1 ms','neural_units_1ms_v1','sitl')]
    rows=[];identity={}
    for case in CASES:
        with np.load(root/'neural_v1'/(case+'_legacy')/'trace.npz') as a,np.load(root/'neural_units_v1'/(case+'_legacy')/'trace.npz') as b:
            identity[case]=all(np.array_equal(a[k],b[k]) for k in ('state','motors','pwm','network_pwm','coordinated_pwm'))
        for label,series,backend in sets:
            folder=root/series/(case+'_'+backend)
            r=json.loads((folder/'report.json').read_text())
            with np.load(folder/'trace.npz') as a:
                changed=abs(a['network_pwm']-a['coordinated_pwm'])>1e-6
                r['xy_changed_fraction']=float(changed.any(axis=1).mean())
                r['xy_changed_by_channel']=changed.mean(axis=0).tolist()
            r['label']=label;rows.append(r)
    result=dict(scope='Ideal synchronous sensors, direct runtime; no deployment admission',
                legacy_exact_between_native_and_units=identity,results=rows,
                numerical_verification=json.loads((root/'neural_units_1ms_v1/validation.json').read_text()))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Порівняння контрактів команд SITL','',
        '| Сцена | Контур | Контакт | Зазор см | Висота max см | STOP | Таймаути | XY змінив команду, % | Фазова власність, % |',
        '|---|---|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['label']} | {r['collision'] or '—'} | {100*r['min_clearance_m']:.2f} | {100*r['max_height_error_m']:.2f} | {r['stop_passed']} | {r['timeout_events']} | {100*r['xy_changed_fraction']:.1f} | {100*r['coordinator_owned_fraction']:.1f} |")
    lines+=['','XY зміна — фактична різниця network/coordinated. Фазова власність — маска phased_owned_channels; вона не включає native STOP.','',
            'Ідентичні legacy траси: '+str(all(identity.values())),
            'Чисельна перевірка: '+json.dumps(result['numerical_verification'])]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(4,3,figsize=(15,14),constrained_layout=True)
    colors=['tab:blue','tab:red','tab:green','black']
    for i,case in enumerate(CASES):
        manifest=json.loads((root/'neural_units_v1'/(case+'_sitl')/'manifest.json').read_text())
        for b in manifest['scene']['boxes']:
            axes[i,0].add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
        for j,(label,series,backend) in enumerate(sets):
            with np.load(root/series/(case+'_'+backend)/'trace.npz') as a:
                s,t=a['state'],a['time_s'];style='--' if j==3 else '-'
                axes[i,0].plot(s[:,0],s[:,1],style,color=colors[j],label=label)
                axes[i,0].scatter(*s[-1,:2],color=colors[j],s=15)
                axes[i,1].plot(t,100*(s[:,2]-1),style,color=colors[j],label=label)
                axes[i,2].plot(t,np.linalg.norm(s[:,3:5],axis=1),style,color=colors[j],label=label)
        axes[i,0].set(title=case,xlabel='X (m)',ylabel='Y (m)',xlim=(-.4,4),ylim=(-1.5,2.5))
        axes[i,0].set_aspect('equal');axes[i,0].scatter(0,0,color='green',marker='s',s=20)
        axes[i,1].set(title='Height error (cm)',xlabel='Model time (s)')
        axes[i,2].set(title='Horizontal speed (m/s)',xlabel='Model time (s)')
        axes[i,2].axhline(.03,color='.5',linestyle=':')
        for ax in axes[i,1:]:ax.axvspan(17,20,color='.5',alpha=.1)
        for ax in axes[i]:ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=8)
    fig.suptitle('Same network and XY controller | different RC interpretation\n'
                 'Native RC collision is with ceiling, not the plotted box; grey time interval = STOP request')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(legacy_exact=all(identity.values()),**result['numerical_verification']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/sitl_bringup_v1'))
    run(p.parse_args().root)
