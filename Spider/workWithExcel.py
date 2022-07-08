
import xlwt
workbook = xlwt.Workbook(encoding="utf-8")  # 创建workbook对象
worksheet = workbook.add_sheet('sheet1')  # 创建工作表
# worksheet.write(0, 0, 'hello')  # 写入数据，第一个行，第二个列，第三个参数
# workbook.save('student.xls')  # 保存数据表单

for i in range(10):
    for g in range(i, 10, 1):
        worksheet.write(i, g, str(i+1)+"*"+str(g+1)+"="+str((i+1)*(g+1)))
workbook.save('calculate.xls')
