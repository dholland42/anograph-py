# `anograph-py`


This is a [`cython`](https://github.com/cython/cython) implementation of the [AnoGraph algorithms](https://github.com/Stream-AD/AnoGraph)
laid out in [Sketch-Based Anomaly Detection in Streaming Graphs](https://dl.acm.org/doi/pdf/10.1145/3580305.3599504)
_Siddharth Bhatia, Mohit Wadhwa, Kenji Kawaguchi, Neil Shah, Philip S. Yu, Bryan Hooi. KDD, 2023_.


## Examples

```python
from anograph.countmin import CountMin, HCountMin

cm = CountMin(2, 32)
cm.add(5)
assert cm.getcount(5) == 1
cm.add_batch([5]*100)  # can be a numpy array
assert cm.getcount(5) == 101

hcm = HCountMin(2, 32)
hcm.add(1, 2)
assert cm.getcount(1, 2) == 1
cm.add_batch([[1, 2]]*100)  # can be a numpy array
assert cm.getcount(1, 2) == 101
```

## Citation

```bibtex
@inproceedings{bhatia2023anograph,
    title={Sketch-Based Anomaly Detection in Streaming Graphs},
    author={Siddharth Bhatia and Mohit Wadhwa and Kenji Kawaguchi and Neil Shah and Philip S. Yu and Bryan Hooi},
    booktitle={SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
    year={2023}
}
