from numpy.typing import ArrayLike

class CountMin:
    def __init__(self, height: int, width: int) -> None:
        """A standard CountMin sketch."""
        ...

    def getcount(self, idx: int) -> float:
        """Get the number of times `idx` has been seen."""
        ...

    def add(self, idx: int) -> None:
        """Add an index into the counter."""
        ...

    def add_batch(self, ArrayLike) -> None:
        """Add a batch of indices to the counter (more efficient)."""
        ...

    def clear(self) -> None:
        """Zero out all counts."""
        ...

class HCountMin:
    def __init__(self, height: int, width: int) -> None:
        """A Higher-Dimension CountMin sketch allowing for counting source, target edges."""
        ...

    def getcount(self, source_idx: int, target_idx: int) -> float:
        """Get the number of times `(source_idx, target_idx)` has been seen."""
        ...

    def add(self, source_idx: int, target_idx: int) -> None:
        """Add a `(source_idx, target_idx)` pair into the counter."""
        ...

    def add_batch(self, ArrayLike) -> None:
        """Add a batch of `(source_idx, target_idx)` pairs to the counter (more efficient)."""
        ...

    def clear(self) -> None:
        """Zero out all counts."""
        ...
