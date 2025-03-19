from pathlib import Path
from multiprocessing import cpu_count
from pprint import pprint
import typer
from ray.data import read_csv
from ray.data.aggregate import Quantile

from anograph import AnoGraph


class Detector:
    def __init__(self, threshold: float | None = None):
        self.ag = AnoGraph(5, 200)
        self.threshold = threshold

    def __call__(self, batch):
        ts = set(batch['timestamp'])
        assert len(ts) == 1
        self.ag.clear()
        self.ag.add_batch(batch['source'], batch['target'])
        score = self.ag.score()
        return dict(score=[float(score > self.threshold) if self.threshold else score], timestamp=list(ts))


def main(data: Path, threshold: float = 50, window: int = 30):
    ds = read_csv(str(data.absolute()))
    ds = ds.map_batches(lambda x: {**x, 'timestamp': x['timestamp']//window})
    grouped = ds.groupby('timestamp')
    scores = grouped.map_groups(Detector, concurrency=min(6, cpu_count()//2), fn_constructor_kwargs=dict(threshold=None))
    qfunc = Quantile(on='score', q=.95, alias_name='p95')
    p95 = scores.select_columns('score').aggregate(qfunc)['p95']
    print(p95)
    pprint(scores.filter(expr=f"score >= {p95}").take_all())
    df = scores.to_pandas()
    outpath = data.parent / 'predictions.csv'
    df.sort_values(by='timestamp').set_index('timestamp').astype(int).to_csv(outpath, index=False, header=False)


if __name__ == '__main__':
    typer.run(main)

