import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

file_path = Path("src/harshu_ai_os/kernel/runtime_events.txt")

runtime_text = file_path.read_text()
logging.info("Runtime event file loaded")

def extract_runtime_values(pattern, runtime_text):
    values = re.findall(pattern, runtime_text)
    return values

module_pattern = r"module=(\w+)"
status_pattern = r"status=(\w+)"
code_pattern = r"code=(\d+)"

module_values = extract_runtime_values(module_pattern, runtime_text)
status_values = extract_runtime_values(status_pattern, runtime_text)
code_values = extract_runtime_values(code_pattern, runtime_text)

print(f"Modules: {module_values}")
print(f"Statuses: {status_values}")
print(f"Codes: {code_values}")


for status in status_values:
    if status == "inactive":
        logging.warning(f"Inactive status found: {status}")

for code in code_values:
    if code == "503":
        logging.error(f"Error code found: {code}")

