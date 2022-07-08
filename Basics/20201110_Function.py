# Function（函数）
def function_sample():
    pass


def print_personal_info():
    print("-----------------------")
    print("|         澂冰         |")
    print("-----------------------")


print_personal_info()


# 带参数的函数
def addition(a, b):
    c = a + b
    print(c)


addition(5, 2)


# 带返回值的函数
def multiply(a, b):
    return a * b


print(multiply(5, 6))


# 返回多个值的函数
def divid(a, b):
    quotient = a//b
    remainder = a % b
    return quotient, remainder


quo1, re1 = divid(5, 2)  # 需要多个值来保存
print("商：%d，余数：%d" % (quo1, re1))

def print_base_line():
    print("-"*30)
def print_multiple_line(a):
    i = 0
    while i < a:
        i += 1
        print_base_line()


print_multiple_line(2)

# 全局变量局部变量名称冲突：有局部变量优先使用；如果没有同名局部变量使用全局

# 在局部区分全局变量
a = 100
def function2():
    global a  # 声明全局变量标识符
    print("This a test: %d"%a)
    a = 200
    print("After: %d"%a)


function2()
print(a)  # 全局变量a已经被修改
