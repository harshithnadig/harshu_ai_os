system_name = "Harshu AI OS"
major_version = 1
release_version = 0.1
debug_mode = True
kernel_enabled = True
maintenance_mode = True

print(f"Welcome to {system_name} v{major_version}.{release_version}")

menu_running = True
while menu_running:
    user_input = input("Please enter (add,subtract,status,boot):(Use skip to skip the current iteration and exit to exit the loop): ")
    if user_input =="add":
        first_number = int(input("Please enter first number: "))
        second_number = int(input("Please enter second number: "))
        addition = first_number + second_number
        print(f"The sum of {first_number} and {second_number} is {addition}")
    elif user_input =="subtract":
        first_number = int(input("Please enter first number: "))
        second_number = int(input("Please enter second number: "))
        subtract = first_number - second_number
        print(f"The difference of {first_number} and {second_number} is {subtract}")
    elif user_input == "status":
        print(f"Debug mode is enabled: {debug_mode}")
        print(f"Kernel mode is enabled: {kernel_enabled}")
        print(f"Maintenance mode is enabled: {maintenance_mode}")
    elif user_input == "boot":
        for boot_step in range(1, 6):
            print(f"{system_name} boot check {boot_step} complete")
    elif user_input == "skip":
        print("Skipping the current iteration")
        continue
    elif user_input == "exit":
        print("Exiting the program")
        break
    else:
        print("Command is invalid")
    
