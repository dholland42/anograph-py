import timeit
from math import sqrt


# Benchark params
NUM_INSERTS = 10_000
ITERATIONS = 10_000

SETUP = f"""
import numpy as np
from countmin import CountMin
cm = CountMin(5, 2_000)
data = np.arange({NUM_INSERTS})
""".strip()

STMT = """
cm.add_batch(data)
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
    print(f"MIN:{minval}\nMAX:{maxval}\nAVG:{meanval} (STD: {stdval})")

    estimate = (minval * 54_000_000_000, maxval * 54_000_000_000)
    print(f"Estimated time for 54B inserts: ({estimate[0]/60/60} - {estimate[1]/60/60}) hours")


if __name__ == "__main__":
    main()

