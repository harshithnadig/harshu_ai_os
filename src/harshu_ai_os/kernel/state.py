import json

json_path = "src/harshu_ai_os/kernel/contacts.json"

contacts = []

try:
    with open(json_path, "r") as file:
        contacts = json.load(file)
except FileNotFoundError:
    print("No saved contacts file found. Starting fresh.")
except json.JSONDecodeError:
    print("Error decoding JSON. File may be corrupted.")
except Exception as error:
    print(f"An error occurred: {error}")

menu_running = True
while menu_running:
    user_input = input("1 → show all contacts\n2 → add a new contact using input()\n3 → search contact by name\n4 → save contacts to JSON\n5 → exit:\n")
    if user_input == "1":
        if len(contacts) == 0:
            print("No contacts found")
        else:
            for contact in contacts:
                print(f"Name: {contact['name']}, Phone: {contact['phone']}, Relation: {contact['relation']}")
    elif user_input == "2":
        user_name = input("Enter contact name: ")
        user_phone = input("Enter contact phone: ")
        user_relation = input("Enter relation: ")
        contacts.append({"name": user_name, "phone": user_phone, "relation": user_relation})
        print(f"Contact added: {user_name}")
    elif user_input == "3":
        user_answer = input("Enter contact name to search: ")
        found = False
        for contact in contacts:
            if contact["name"].lower() == user_answer.lower():
                print(contact)
                found = True
                break
        if found == False:
            print("Contact not found")
    elif user_input == "4":
        with open(json_path, "w") as file:
            json.dump(contacts, file, indent=2)
        print("Contacts saved to JSON")
    elif user_input == "5":
        with open(json_path, "w") as file:
            json.dump(contacts, file, indent=2)
        print("Exiting and saving contacts")
        break
    else:
        print("Invalid input")