import time
# Handle Exception
try:
    print("Before")
    f = open("test1.txt", "r")
    print("After")
    f.close()
except IOError:
    pass

try:
    pass
except (NameError, IOError):  # Can handle multiple exceptions
    pass

try:
    pass
except (NameError, IOError) as result:  # 相当于Java的printExceptionMessage
    print(result)

try:
    pass
except Exception as result:  # 捕获所有异常
    print(result)

# try catch finally

try:
    pass
except Exception as result:
    pass
finally:  # Finally will execute no matter exception exist
    pass

try:
    try:
        f = open("test.txt", "r")
        while True:
            content = f.readline()
            if len(content) == 0:
                break
            time.sleep(0.1)
            print(content)
    finally:
        f.close()
except Exception as result:
    print(result)

try:
    try:
        f = open("test.txt", "r")
        f2 = open("poem.txt", "w")
        while True:
            content = f.readline()
            if len(content) == 0:
                break
            time.sleep(0.1)
            f2.write(content)
    finally:
        f.close()
        f2.close()
except Exception as result:
    print(result)