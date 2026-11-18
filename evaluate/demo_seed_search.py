"""Find demo seeds for the README (step 8): exactly the scene the browser demo builds.

Scene: random_room, 20 % density, layout clustered or uniform, start (-0.7, 0, 1), yaw 0, 60 s, AUTO at 0.
For each seed and layout both phased_side_guard and pipeline_v1 fly once (Standalone). Reported per run:
contact, minimum clearance, network share (the demo's live indicator) and the share of phase FAILED.
This only picks illustrative seeds; it is not an evaluation and tunes nothing.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json
from pathlib import Path


def fly(seed, layout, profile, seconds=60.):
    from demo.live_view import running_share
    from demo.runner import DemoRunner
    with closing(DemoRunner(controller_mode=profile, scene='random_room', seed=seed, start=(-.7, 0., 1.), yaw=0.,
                            scene_options=dict(obstacle_density_pct=20, layout_style=layout),
                            max_duration_s=seconds)) as r:
        r.resume()
        r.event(dict(type='mode', mode='AUTO'))
        phases = Counter()
        while not r.terminal:
            r.advance(5)
            phases[r.system.runtime.coordinator.last_decision.get('phased_phase')] += 1
        share, ticks = running_share(r)
        clearance = min(row['clearance_m'] for row in r.rows)
        n = sum(phases.values())
        return dict(seed=seed, layout=layout, profile=profile, contact=r.rows[-1]['collision'], duration_s=r.time_s,
                    min_clearance_m=clearance, network_share=share, failed_share=phases.get('FAILED', 0)/n if n else 0.,
                    turn_share=phases.get('TURN', 0)/n if n else 0.)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs=2, default=[20261021, 20261041])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    tasks = [(s, layout, prof) for s in range(*a.seeds) for layout in ('clustered', 'uniform')
             for prof in ('phased_side_guard', 'pipeline_v1')]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(fly, *zip(*tasks)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(len(results), 'runs')
