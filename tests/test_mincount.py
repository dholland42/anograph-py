import numpy as np
from anograph import HCM, u64_hash, vhash, AnoGraph

def test_ano_density():
    NUM_BUCKETS = 2
    hcm = HCM(NUM_BUCKETS)
    hcm.add(1, 2)
    assert hcm.sketch.sum() == NUM_BUCKETS
    hcm.add(1, 2)
    assert hcm.sketch.sum() == 2 * NUM_BUCKETS
    hcm.clear()
    assert hcm.sketch.sum() == 0

def test_u64_hash():
    assert u64_hash("some node data") > 0

def test_vectorized_hash():
    data = np.array([f"node_{i}" for i in range(100)]).reshape((-1, 2))
    assert np.all(vhash(data) > 0)

def test_add_batch():
    hcm = HCM()
    source = np.array([f"node_{i}" for i in range(32)])
    target = np.array([f"node_{i}" for i in range(32)])
    np.random.shuffle(target)
    hcm.add_batch(source, target)
    assert np.sum(hcm.sketch) == 64

def test_ano():
    ag = AnoGraph()
    assert ag.score() == 0
    source = np.array([f"node_{i}" for i in range(32)])
    target = np.array([f"node_{i}" for i in range(32)])
    np.random.shuffle(target)
    ag.add_batch(source, target)
    assert ag.score() != 0

