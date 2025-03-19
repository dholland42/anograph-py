# `anograph-py`


This is a [`cython`](https://github.com/cython/cython) implementation of the [AnoGraph algorithms](https://github.com/Stream-AD/AnoGraph)
laid out in [Sketch-Based Anomaly Detection in Streaming Graphs](https://dl.acm.org/doi/pdf/10.1145/3580305.3599504)
_Siddharth Bhatia, Mohit Wadhwa, Kenji Kawaguchi, Neil Shah, Philip S. Yu, Bryan Hooi. KDD, 2023_.


## Examples

```python
import numpy as np
from anograph import AnoGraph

ag = AnoGraph(num_buckets=2, bucket_size=32)
ag.add(1, 2)
ag.add_batch(np.random.randint(size=(100,2), dtype=np.uint64))

# It is presumed that hashing is done beforehand, so generate
# hashes if you need to.
source, target = "some source", "some target"
ag.add(hash(source), hash(target))
```

## Citation

```bibtex
@inproceedings{bhatia2023anograph,
    title={Sketch-Based Anomaly Detection in Streaming Graphs},
    author={Siddharth Bhatia and Mohit Wadhwa and Kenji Kawaguchi and Neil Shah and Philip S. Yu and Bryan Hooi},
    booktitle={SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
    year={2023}
}
