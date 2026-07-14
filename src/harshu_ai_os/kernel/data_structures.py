from collections import deque

task_stack = []
task_stack.append("wash clothes")
task_stack.append("buy groceries")
task_stack.append("charge laptop")
print(task_stack)

if task_stack:
    removed_task = task_stack.pop()
    print(removed_task)
else:
    print("Stack is empty")

print(task_stack)

task_queue = deque()

task_queue.append("wash clothes")
task_queue.append("buy groceries")
task_queue.append("charge laptop")

print(task_queue)

if task_queue:
    removed_task_queue = task_queue.popleft()
    print(removed_task_queue)
else:
    print("Queue is empty")

print(task_queue)