"""
# 打开文件，默认是r（只读）模式
f1 = open("test.txt", "w")  # w是写入模式，不存在则新建，存在则覆盖
# rb 是二进制只读模式； wb是二进制写入模式

# 写入
f1.write("Hello World!")

# 关闭文件
f1.close()
"""

"""
# Read方法，读取指定字符，开始时定位在文件头部，每执行一次向后移动指定字符数
f2 = open("test.txt", "r")
content = f2.read(5)  # Read 5 char
print(content)
content2 = f2.read(5)  # Read another 5 char
print(content2)

f2.close()
"""

"""
# readlines()以列表方式读取全部内容，一行成为一个字符串元素
f3 = open("test.txt", "r")
content3 = f3.readlines()
print(content3)

i = 1
for temp in content3:
    print("%d:%s" % (i, temp))
    i += 1
f3.close()
"""

# readline()读取一行
f4 = open("test.txt", "r")
content4 = f4.readline()
print("1:%s" % content4)
content4 = f4.readline()
print("2:%s" % content4)
f4.close()

# 重命名和删除
import os
os.rename("target.txt", "renamed.txt")
os.remove("target.txt")

