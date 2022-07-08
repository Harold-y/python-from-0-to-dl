
# 查询: in or not in
list1 = ["New York", "Shanghai", "Tokyo"]
# findCity = input("please insert: ")
findCity = "Vancouver"
if findCity in list1:
    print("Result found! ")
else:
    print("Match not found. ")

list2 = ["a", "b", "c", "d", "a", "s", "d"]
print(list2.index("a", 0, 5))  # 后两个index指定范围：从x开始，到y结束（不包括y）
print(list2.index("a", 1, 5))  # 如果有两个结果返回第一个；如果没有找到则报错
print(list2.count("d"))  # 统计某元素出现几次

# 排序
list3 = [1, 5, 62, 24, 632, 4]
print(list3)
list3.reverse()  # 将列表反转
print(list3)
list3.sort()  # 排序， 升序
print(list3)
list3.sort(reverse=True)  # 排序， 降序
print(list3)

# 二维数组和嵌套
product = [[], [], []]  # 三个元素的空列表，每个元素都为列表
product = [["Core I9", "Core I7", "Core I5"], ["Ryzen 5", "Ryzen 7", "Ryzen 3"],
           ["Aurora 12", "Aurora 10", "Aurora 5", "Aurora 8"]]
print(product[0])
print(product[2][0])

