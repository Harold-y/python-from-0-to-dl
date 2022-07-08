if True:
    print("True")
else:
    print("False")

if 100:
    print("100, Representative for True")

if 0:
    print("0 Represent for False")
else:
    print("0, representative for False")

score = 87
if score >= 90 and score <= 100:
    print("A")
elif score >= 80 and score < 90:
    print("B")
else:
    print("E")

import random  # import random
s = random.randint(0, 100)  # auto generate a random number of Integer from 0 to 100
# inclusively
print(s)

#Homework
userInput = int(input("剪刀，石头，布 - 0, 1, 2, respectively:  "))
AIChoice = random.randint(0, 2)
print(userInput)
print(AIChoice)

if userInput == 0:
    if AIChoice == 0:
        print("Pair")
    if AIChoice == 1:
        print("AI Win!")
    if AIChoice == 2:
        print("You win!")
elif userInput == 1:
    if AIChoice == 1:
        print("Pair")
    if AIChoice == 2:
        print("AI Win!")
    if AIChoice == 0:
        print("You win!")
elif userInput == 2:
    if AIChoice == 2:
        print("Pair")
    if AIChoice == 0:
        print("AI Win!")
    if AIChoice == 1:
        print("You win!")



