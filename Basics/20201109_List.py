
# 空列表
nameList = []

# List可以用混合式的填充，保持原有类型
list1 = ['abcd', 242, 2.52]
print(type(list1[0]))
print(type(list1[1]))

# 索引值可以从0开始，-1为从末尾开始
# 列表可以使用+进行拼接，用*重复显示
list2 = [123, "Seattle"]
print(list1+list2)

# 下标使用
nameList = ["Harold Finch", "John Reese", "Joss Carter", "Lionel Fusco"]
print(nameList[0])
print(nameList[3])
print(nameList[-1])  # read the last one

print("-"*30)  # Clear Console

# 长度
print(len(nameList))

# 循环打印列表
for name in nameList:
    print(name)

i = 0
while i < len(nameList):
    print(nameList[i])
    i += 1

'''
list1.append(5) 增加数据到尾部
list1.extend(list2) 追加
list1.insert(1,3) 插入
del list1[0] 删除指定index元素
list1.remove(1) 查找list1第一个匹配int 1的值，如果找到则移除，没有找到抛出异常
list1.pop() 删除尾部元素
list1.sort() 排序
list1.reverse() 反转

列表切片使用[: :] e.g.: list1[2:5:2]
'''

# 增加
print(nameList)
nameList.append("The Machine")
print("-"*30)
print(nameList)

print("-"*30)
nameList.insert(0, "Root")
print(nameList)

print("-"*30)

# 追加列表
a = [1, 2]
b = [3, 4]
c = [5, 6]
a.append(b)  # 二位嵌套，把b作为一个元素加入a
print(a)

b.extend(c)  # 扩展，用extend; 将c的元素一个一个加入b中
print(b)

# 删除
print("-"*30)
d = [0, 1, 2, 3, 4, 2, 5]
print(d)

del d[0]
print(d)

# d.remove(6)  # 报错
d.remove(2)
print(d)

d.pop()
print(d)

# 改变数组
print("-"*30)
e = [2, 5, 3, 5]
index = input("修改的数字：")
e[0] = index
print(e)
