"""Controlled data-structure benchmarks retained from the Python foundation."""

from time import perf_counter


def measure_membership(
    data_size: int,
    repeat_count: int = 1_000,
) -> dict[str, float]:
    """Compare equivalent membership checks after setup is complete."""
    number_list = list(range(data_size))
    number_set = set(range(data_size))
    number_dictionary = {number: True for number in range(data_size)}
    target = data_size - 1

    results = {}
    for name, collection in (
        ("list", number_list),
        ("set", number_set),
        ("dictionary", number_dictionary),
    ):
        start_time = perf_counter()
        for _ in range(repeat_count):
            target in collection
        results[name] = perf_counter() - start_time

    return results


def count_quadratic_operations(size: int) -> int:
    """Make n-squared growth visible without relying only on noisy timings."""
    operation_count = 0

    for _ in range(size):
        for _ in range(size):
            operation_count += 1

    return operation_count


def run_benchmarks() -> None:
    """Print the controlled foundation benchmark for manual observation."""
    for data_size in (1_000, 5_000, 10_000):
        results = measure_membership(data_size)
        print(f"Data size: {data_size}")
        for name, elapsed in results.items():
            print(f"{name.title()}: {elapsed:.6f} seconds")
        print()

    print("O(n²) nested-loop test")
    for size in (100, 200, 400):
        print(f"Size: {size}, Operations: {count_quadratic_operations(size)}")


if __name__ == "__main__":
    run_benchmarks()
