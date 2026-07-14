runtime_state = {
    "boot_status": "complete",
    "active_users": 2,
    "restart_count": 1,
    "last_health_check": "passed"
}

user_count = input("Please enter active user count: ")

try:
    user_count = int(user_count)
    print(f"Active users: {user_count}")
except ValueError:
    print("Invalid input. Please enter a number.")

restart_count = input("Please enter the restart count: ")

try:
    restart_count = int(restart_count)
    system_score = 100 / restart_count
    print(f"System score: {system_score}")
except ValueError:
    print("Restart count must be a number.")
except ZeroDivisionError:
    print("Restart count cannot be zero.")

class RuntimeStateError(Exception):
    pass

def validate_runtime_state(runtime_state):
    if runtime_state['boot_status'] == "":
        raise RuntimeStateError("boot_status cannot be empty")
    elif runtime_state['active_users'] < 0:
        raise RuntimeStateError("active_users cannot be negative")
    elif runtime_state['restart_count'] <= 0:
        raise RuntimeStateError("restart_count must be greater than 0")
    else:
        return True

try:
    validate_runtime_state(runtime_state)
    print(f"Validation Passed")
except RuntimeStateError as error:
    print(f"Validation failed: {error}")
finally:
    print("All validation checks finished")

print(runtime_state['boot_status'])
print(runtime_state['active_users'])
print(runtime_state['restart_count'])
print(runtime_state['last_health_check'])