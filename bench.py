import timeit
from math import sqrt


# Benchark params
NUM_INSERTS = 10_000
ITERATIONS = 100

SETUP = f"""
import numpy as np
from anograph import HCM
hcm = HCM(2, 32)
data = np.array([f"node_{{i}}" for i in range({NUM_INSERTS})])
""".strip()

STMT = """
hcm.add_batch(data, data)
""".strip()

# Second defs for formatting
S = 1
MS = S / 1_000
NS = MS / 1_000


def format_val(x: float) -> str:
    if x > S:
        return f"{x:.2f}s"
    elif x > MS:
        return f"{x*1000:.2f}ms"
    else:
        return f"{x*1000*1000:.2f}ns"


def main():
    timer = timeit.Timer(stmt=STMT, setup=SETUP)
    results = timer.repeat(repeat=5, number=ITERATIONS)
    results = [x / ITERATIONS / NUM_INSERTS for x in results]
    minval = min(results)
    maxval = max(results)
    meanval = sum(results) / len(results)
    stdval = sqrt(sum(((x - meanval)**2 for x in results)) / (len(results) - 1))
    print(f"MIN:{format_val(minval)}\nMAX:{format_val(maxval)}\nAVG:{format_val(meanval)} (STD: {format_val(stdval)})")

    estimate = (minval * 54_000_000_000, maxval * 54_000_000_000)
    print(f"Estimated time for 54B inserts: ({estimate[0]/60/60} - {estimate[1]/60/60}) hours")

    estimate = (minval * 200_000_000, maxval * 200_000_000)
    print(f"Estimated time for 200M inserts: ({estimate[0]/60/60} - {estimate[1]/60/60}) hours")


if __name__ == "__main__":
    main()

