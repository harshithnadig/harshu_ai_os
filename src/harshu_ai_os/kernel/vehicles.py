class Vehicle:
    def __init__(self,brand):
        self.brand = brand
    def move(self):
        print("Vehicle is moving")

class Car(Vehicle):
    def __init__(self,brand,doors):
        super().__init__(brand)
        self.doors = doors
    def move(self):
        print("Car is driving")
    def __str__(self):
        return f"Brand: {self.brand}, Doors: {self.doors}"

class Truck(Vehicle):
    def __init__(self,brand,load_capacity):
        super().__init__(brand)
        self.load_capacity = load_capacity

    def move(self):
        print("Truck is transporting goods")

    def __str__(self):
        return f"Brand: {self.brand}, Load capacity: {self.load_capacity}"


vehicle = Vehicle("Generic")
vehicle.move()

car = Car("Toyota", 4)
car.move()
print(car)


truck = Truck("Tata", 5000)
print(truck)

truck.move()
