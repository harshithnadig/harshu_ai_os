"""Small interactive CLI retained as the local Harshu AI OS entry point."""

from harshu_ai_os.kernel.runtime import RuntimeProfile, get_app_mode


def read_numbers() -> tuple[int, int]:
    """Read the two integer operands shared by arithmetic commands."""
    first_number = int(input("Please enter first number: "))
    second_number = int(input("Please enter second number: "))
    return first_number, second_number


def run_cli() -> None:
    """Run the local command loop without starting it during module imports."""
    profile = RuntimeProfile(
        system_name="Harshu AI OS",
        mode=get_app_mode(),
    )
    print(profile.show_summary())

    while True:
        command = (
            input("Enter add, subtract, status, boot, skip, or exit: ").strip().lower()
        )

        if command == "add":
            first_number, second_number = read_numbers()
            print(
                f"The sum of {first_number} and {second_number} is "
                f"{first_number + second_number}"
            )
        elif command == "subtract":
            first_number, second_number = read_numbers()
            print(
                f"The difference of {first_number} and {second_number} is "
                f"{first_number - second_number}"
            )
        elif command == "status":
            print("Kernel enabled: True")
            print(f"Mode: {profile.mode}")
        elif command == "boot":
            for boot_step in range(1, 6):
                print(f"{profile.system_name} boot check {boot_step} complete")
        elif command == "skip":
            print("Skipping the current iteration")
            continue
        elif command == "exit":
            print("Exiting the program")
            break
        else:
            print("Command is invalid")


if __name__ == "__main__":
    run_cli()
