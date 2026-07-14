import time

def measure_runtime(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime: {runtime} seconds")
        return result
    return wrapper
    
@measure_runtime
def show_module_status(module_name):
    return f"Module status checked: {module_name}"

status_message = show_module_status("config")
print(status_message)