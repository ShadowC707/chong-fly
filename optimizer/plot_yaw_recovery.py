"""Plot paired recovery experiments without recomputing predictions or training."""
import argparse
import json
from pathlib import Path

from artifacts.io import publication, write_json
from generator.source_contract import file_sha256


def plot_report(report_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['version'] != 'yaw-recovery-paired-v1':
        raise ValueError('Expected paired yaw recovery report')
    with publication(output) as stage:
        for candidate in report['candidates']:
            versions = {'initial': candidate['initial'], **candidate['arms']}
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained', sharex=True)
            for ax, (side, width) in zip(axes.flat, [('left', 1), ('right', 1), ('left', 4), ('right', 4)]):
                for name, value in versions.items():
                    probe = value['recovery']
                    row = next(r for r in probe['cases'] if r['side'] == side
                               and r['width'] == width and r['normalized_distance'] == .1)
                    time = (np.arange(len(row['yaw_pwm']))-probe['warmup_steps'])*probe['dt']
                    ax.plot(time, row['yaw_pwm'], label=name)
                    if name == 'initial':
                        ax.plot(time, row['teacher_yaw_pwm'], '--', color='black', label='teacher v5')
                        ax.axvspan(0, probe['threat_steps']*probe['dt'], color='orange', alpha=.13)
                        ax.axhspan(1480, 1520, color='green', alpha=.10)
                        ax.axvline(probe['threat_steps']*probe['dt']+.8, color='gray', linestyle=':')
                ax.set(title=f'{side} obstacle | {width} column(s) | 0.35 m',
                       xlabel='Seconds from threat onset', ylabel='Yaw command (PWM)')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle(candidate['candidate']+' | 10-epoch paired continuation | synthetic sensor probes')
            fig.savefig(stage/(candidate['candidate']+'.png'), dpi=140)
            plt.close(fig)
        write_json(stage/'manifest.json', dict(source_report=str(report_path.resolve()),
            source_sha256=file_sha256(report_path),
            plot_code_sha256=file_sha256(Path(__file__)),
            selection='Both sides, near obstacles, widths 1 and 4; all cases remain in source report.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    plot_report(args.report, args.out_dir)


if __name__ == '__main__':
    main()
