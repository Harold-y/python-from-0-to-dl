# for loop 1
for i in range(5):
    print("我喜欢何同学"+str(i)+"次")

# for loop 2
for i in range(0, 10, 3):  # for loop start from 0, add 3 each time, end <10
    print(i)

for i in range(-10, -100, -3):
    print(i)

name = "Success"
for x in name:
    print(x, end="\t")

a = ["aa", "bb", "cc", "dd"]
for i in range(len(a)):
    print(i, a[i])

