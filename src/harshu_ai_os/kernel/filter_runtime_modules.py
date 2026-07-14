import csv
from pathlib import Path

path = "data/runtime_modules.csv"
active_modules= []
with open(path,'r')as file:
        new_data =csv.DictReader(file)
        for data in new_data:
            if data['status'] == 'active':
                 active_modules.append(data)


output = Path("data/active_runtime_modules.csv")

with open(output,'w',newline="")as file:
    writer = csv.DictWriter(file,fieldnames=["name", "status", "health_score"])
    writer.writeheader()
    writer.writerows(active_modules)

print(f"Saved {len(active_modules)} active modules to {output}")
