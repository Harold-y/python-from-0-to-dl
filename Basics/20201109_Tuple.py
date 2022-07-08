
# Tuple本身元素不可改变，但是里面的子元素可以改变
tup1 = ()  # 空Tuple
tup2 = (50)  # 整型而非元组
tup3 = (50,)  # 元组类型
tup4 = (50, 25, 521)  # 元组类型
print(type(tup1))
print(type(tup2))
print(type(tup3))
print(type(tup4))

# 访问
tup5 = ("abx", "gwe", 1265, 1256.2, 245.2, 6.2)
print(tup5[0])
print(tup5[-1])  # get last element, just like did in List
print(tup5[1:5])  # 切片，左闭右开（不包含后者）

# 增
graphicCard1 = ("GTX1080TI", "RTX2060SUPER")
graphicCard2 = ("RX5700XT", "RX5900XT")
graphicCard = graphicCard1 + graphicCard2  # Create a new element, connecting two
# existing
print(graphicCard)

# 删
del graphicCard  # 删除整个元组变量
# print(graphicCard)  # 报错，因为元组已经被删除


# 改
# graphicCard = ("GTX1080TI", "RTX2060SUPER")
# graphicCard[0] = "RTX3070"  # 报错：不允许修改

# 查


# 变换成元组
# tuple(otherType)
