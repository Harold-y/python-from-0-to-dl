
# Python 3 default UTF-8 and all Strs are unicode
# 支持字符串拼接，用\转义等等
# 三引号可以保存所有格式
# "" is recommended for normal use

a = "Hello"
b = 'Python'
para = """
        This is a paragraph
        Can be made by multiple lines
        
        Snow moving down from sky of Wisconsin
        My heart goes forever with the cold of Madison
"""

print(a)
print(b)
print(para)

# 极少情况下 ""和''不一样
'''
\n换行
\t缩进
\\ 正常化\
\" 正常化"
\' 正常化'
'''
print("I'm a student")  # Can not be done with NORMAL ''
print('I\'m a student')  # \ is needed for ''
print("Harold said \"I made the Machine\"")  # "" also need \ for inner ""

# String array can be used to partially get content
# [Start Position : End Position : 步进值]
city = "Vancouver is in BC, Canada. "
city2 = "Los Angeles, CA, USA"
print(city)
print(city[0:6])
print(city[0])
print(city[1:7:2])
print(city2[1:7:2])
print(city[2:])  # From the GIVEN starting point to the end
print(city[:10])  # From the first to the GIVEN end position

print(city+"Welcome to Canada, my Chinese guest!")
print(city*3)  # Print multiple time

print("Hello\nHarold")  # \ for escape （\当作转义行为）
print(r"Hello\nHarold")  # r at front can cancel role of \ （r可以取消转义）

# 字符串常见操作
'''
bytes.decode(encoding="utf-8", errors="strict")
Python没有decode方法，可以使用bytes对象的decode()来解码给定bytes对象，这个bytes对象可以由
str.encode()来编码返回

encode(encoding='UTF-8',errors='strict')
以encoding指定的编码格式编码字符串，如果出错默认报ValueError异常，除非errors指定的是'ignore'或'replace'

isalnum() 如果字符串至少有一个字符串并且所有字符都是字母或者数字则返回True，否则False
isalpha() 字符串至少有一个字符，并且所有字符都是字母返回True，否则False
isdigit() 如果字符串只包含数字返回True，否则False
isnumeric() 如果字符串中只包含数字字符则返回True，否则False

join(seq) 以指定字符串作为分隔符，将seq中所有元素（的字符串表示）合并为一个新的字符串 seq:分隔符
len(string) 返回字符串长度
Istrip() 截掉字符串左边的空格和指定字符
rstrip() 删除字符串末尾的空格
split(str="",num=string.count(str)) num=string.count(str)) 分开字符串
'''