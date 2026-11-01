"""Plot the fixed-workload 1/2/3-owner measurements from the repair study."""
import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--preview', type=pathlib.Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [row for row in json.loads(args.summary.read_text())
            if row['experiment'].startswith('repaired-v2-')
            and row['learners'] in (128, 512)]
    assert {(r['gpu_owners'], r['learners']) for r in rows} == {
        (gpu, level) for gpu in (1, 2, 3) for level in (128, 512)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for level, color in [(128, '#2364A5'), (512, '#D06B24')]:
        points = sorted((r for r in rows if r['learners'] == level),
                        key=lambda r: r['gpu_owners'])
        x = [r['gpu_owners'] for r in points]
        for ax, field, factor in [(axes[0], 'steps_per_second_median', 1),
                                  (axes[1], 'step_p99_ms_median', .001)]:
            y = [r[field] * factor for r in points]
            ax.plot(x, y, color=color, marker='o', linewidth=2,
                    label=f'{level} concurrent Learners')
            for xx, yy in zip(x, y):
                ax.annotate(f'{yy:.1f}', (xx, yy), xytext=(0, -15 if ax is axes[0] and level == 512 else 7),
                            textcoords='offset points', ha='center',
                            fontsize=9, color=color)
    for ax in axes:
        ax.set_xticks([1, 2, 3]); ax.set_xlabel('Physical GPU parameter owners')
        ax.set_xlim(.8, 3.2); ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    axes[0].set_ylim(0, 45); axes[0].set_ylabel('Acknowledged sparse updates / second')
    axes[1].set_ylim(0, 22); axes[1].set_ylabel('Trial P99 end-to-end latency (seconds)')
    axes[0].legend(frameon=False, loc='lower left', fontsize=9)
    fig.suptitle('Large-table scaling on 1–3 RTX A6000 owners', fontsize=15, y=.98)
    fig.text(.5, .895, '1,048,576 × 512 FP32 · four PS shards · 8,192 IDs/update · fixed total work',
             ha='center', fontsize=10, color='#444444')
    fig.text(.5, .025, 'Median of two 5 s submission windows + full drain · 256 MiB ticket payload/shard · BRPC over SSH',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.075, right=.97, top=.81, bottom=.18, wspace=.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={'Date': None})
    if args.out.suffix.lower() == '.svg':
        args.out.write_text('\n'.join(line.rstrip() for line in
                                      args.out.read_text().splitlines()) + '\n')
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
