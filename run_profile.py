import cProfile

import numpy as np

from anograph.countmin import CountMin

cm = CountMin(5, 2_000)
data = np.arange(100_000_000)
cProfile.run("cm.add_batch(data)")

