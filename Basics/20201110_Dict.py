
# dict（字典：键值对）
desktop = {"CPU": "Ryzen 5 5600X", "GPU": "RX5900XT"}
print(desktop["CPU"])
# print(desktop["Memory"])  # 报错，因为访问不存在的键
print(desktop.get("Memory"))  # 不报错，虽然get()没有但是非直接访问
print(desktop.get("Memory", "0G"))  # 可以设置没找到的默认值

# 增
desktop = {"CPU": "Ryzen 5 5600X", "GPU": "RX5900XT"}
newMemory = "8G Samsung DDR4 3222 * 2"
desktop["Memory"] = newMemory
print(desktop)

# 删
# del
print("CPU Before Delete: %s"%desktop["CPU"])
# del desktop["CPU"]  # 删除键和值， 再访问会报错
print("Before delete: %s"%desktop)
# del desktop  # 删除整个字典，再访问会报错

# clear
print("Before clear: %s"%desktop)
desktop.clear()
print("After delete: %s"%desktop)

# 改
desktop = {"CPU": "Ryzen 5 5600X", "GPU": "RX5900XT"}
desktop["GPU"] = "RTX3090"
print(desktop["GPU"])


# 查
desktop = {"CPU": "Ryzen 5 5600X", "GPU": "RX5900XT", "Memory": "8G Samsung DDR4 3222 * 2"}
print(desktop.keys())  # get all keys (returned by list)
print(desktop.values())  # get all values (returned by list)
print(desktop.items())  # get all items（键值对） by List containing Tuple

print("-"*30)

# 遍历所有的键
for key in desktop.keys():
    print(key)

# 遍历所有的值
for value in desktop.values():
    print(value)

# 遍历键值对
for key, value in desktop.items():
    print("key=%s, value=%s" % (key, value))

# 枚举，for循环使用下标
list1 = ["a", "b", "c", "d"]
for i, x in enumerate(list1):
    print(i, x)
