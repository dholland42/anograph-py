import numpy as np

from countmin import CountMin, HCountMin


def test_countmin():
    cm = CountMin(5, 2_000)
    for _ in range(100):
        cm.add(5)
    assert cm.getcount(5) == 100
    cm.add_batch(np.ones(shape=(100,))*5)
    assert cm.getcount(5) == 200
    cm.clear()
    assert cm.getcount(5) == 0
    cm.add_batch(range(100))
    for i in range(100):
        assert cm.getcount(i) > 0

def test_hcountmin():
    hcm = HCountMin(2, 32)
    for _ in range(100):
        hcm.add(1, 2)
    assert hcm.getcount(1, 2) == 100
    hcm.add_batch(np.array([[1, 2]]*100))
    assert hcm.getcount(1, 2) == 200
    hcm.clear()
    assert hcm.getcount(1, 2) == 0
    hcm.add_batch([[i, i + 1] for i in range(100)])
    for i in range(100):
        assert hcm.getcount(i, i + 1) > 0

