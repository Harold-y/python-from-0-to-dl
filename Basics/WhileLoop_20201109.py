# while loop 1 and use of %
i = 0
while i < 5:
    print("It's %d time of loop"%(i+1))
    print("i = %d"%i)
    i = i + 1

# while loop 2: sum from 1 to 100
current = 1
sumNum = 0
countTo = 100
while current <= countTo:
    sumNum = sumNum + current
    current = current + 1
print("From 1 to %d, sum is %d" % (countTo, sumNum))

# while loop 3: collaborate with else
count = 0
while count < 5:
    print(count, "Less than 5")
    count += 1
else:
    print(count, "Greater / Equal to 5")

# break and continue
# break 跳出 for or while
# continue 跳出当前循环轮，进行下一次循环
# pass空语句，占位，不做任何事情

csd = 0
while csd < 10:
    csd += 1
    print("-"*30)
    if csd == 5:
        break
    print(csd)

csd = 0
while csd < 10:
    csd += 1
    print("-"*30)
    if csd == 5:
        continue
    print(csd)
