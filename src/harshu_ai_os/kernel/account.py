from dataclasses import dataclass

@dataclass        
class BankAccount:
    owner:str
    balance:int = 0

    def deposit(self,amount:int) -> int:
        self.balance = self.balance + amount
        return self.balance

    def withdraw(self,amount:int) -> int | str:
        if amount <= self.balance:
            self.balance = self.balance - amount
            return self.balance
        else:
            return "Insufficient balance"

    def show_balance(self) -> str:
        return f"Owner: {self.owner}, Balance: {self.balance}"


if __name__ == "__main__":

    account = BankAccount("Harshu")


    print(account.deposit(500))
    print(account.withdraw(200))
    print(account.withdraw(1000))
    print(account.show_balance())
    print(account)


