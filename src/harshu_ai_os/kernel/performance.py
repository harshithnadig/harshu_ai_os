from time import perf_counter

input_sizes = [1_000, 5_000, 10_000]
repeat_count = 1_000

for data_size in input_sizes:
    # Initialize the data structures before starting the timer
    number_list = list(range(data_size))
    number_set = set(range(data_size))
    number_dictionary = {number: True for number in range(data_size)}
    target = data_size - 1

    # Measure search time in List
    start_time = perf_counter()
    for _ in range(repeat_count):
        target_found = target in number_list
    end_time = perf_counter()
    list_elapsed = end_time - start_time

    # Measure search time in Set
    start_time = perf_counter()
    for _ in range(repeat_count):
        target_found_in_set = target in number_set
    end_time = perf_counter()
    set_elapsed = end_time - start_time

    # Measure search time in Dictionary
    start_time = perf_counter()
    for _ in range(repeat_count):
        target_found_in_dict = target in number_dictionary
    end_time = perf_counter()
    dict_elapsed = end_time - start_time

    # Output results for current data size
    print(f"Data size: {data_size}")
    print(f"List: {list_elapsed:.6f} seconds")
    print(f"Set: {set_elapsed:.6f} seconds")
    print(f"Dictionary: {dict_elapsed:.6f} seconds")
    print()

nested_sizes = [100, 200, 400]
print("O(n²) nested-loop test")
for size in nested_sizes:
    operation_count = 0
    start_time = perf_counter()

    for _ in range(size):
        for _ in range(size):
            operation_count += 1

    elapsed = perf_counter() - start_time

    print(
        f"Size: {size}, "
        f"Operations: {operation_count}, "
        f"Time: {elapsed:.6f} seconds"
    )