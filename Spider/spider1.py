# coding = utf-8

import sys
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，文字匹配
import urllib.request  # 指定url，获取网页数据
import urllib.response
import xlwt  # 进行Excel操作
import _sqlite3  # 进行SQLite数据库操作



def main():
    baseurl = "https://movie.douban.com/top250?start="
    data_list = get_data(baseurl)
    save_path = ".\\Douban Film2.xls"
    save_data(data_list, save_path)
    # askURL("https://movie.douban.com/top250?start=")


# Film Detail URL Rule
findLink = re.compile(r'<a href="(.*?)">')  # 创建正则表达式对象，表示规则（字符串的模式）
# Film Image URL Rule
findImgSrc = re.compile(r'<img.*src="(.*?)"', re.S)  # re.S 忽略换行符

findTitle = re.compile(r'<span class="title">(.*)</span>')
findOtherTitle = re.compile(r'<span class="other">(.*)</span>')
findRatingScore = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
findRatingNumber = re.compile(r'<span>(\d*)人评价</span>')  # \d数字
findInq = re.compile(r'<span class="inq">(.*)</span>')
findDetail = re.compile(r'<p class="">(.*?)</p>', re.S)

# 1 爬取网页
def get_data(baseurl):
    data_list = []
    for i in range(0, 10):
        url = baseurl + str(i * 25)
        html = askURL(url)

        # 2 解析数据
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all('div', class_="item"):  # 查找符合要求的字符串，形成列表
            data = []  # secure all information for a single film
            item = str(item)
            # Film Detail Link URL
            link = re.findall(findLink, item)[0]  # re库用正则表达式来查找指定字符串
            data.append(link)  # append link

            img_link = re.findall(findImgSrc, item)[0]
            data.append(img_link)

            title = re.findall(findTitle, item)
            if len(title) == 2:
                cn_title = title[0]
                data.append(cn_title)  # append Chinese name
                fo_title = title[1].replace("/", "")  # 去掉无关字符/
                data.append(fo_title)  # append Foreign name
            else:
                data.append(title[0])
                data.append('')  # 留空

            other_title = re.findall(findOtherTitle, item)
            data.append(other_title)

            rating = re.findall(findRatingScore, item)[0]
            data.append(rating)

            rating_number = re.findall(findRatingNumber, item)[0]
            data.append(rating_number)

            inq = re.findall(findInq, item)
            if len(inq) != 0:
                data.append(inq[0].replace("。", ""))
            else:
                data.append('')  # 留空

            detail = re.findall(findDetail, item)[0]
            detail = re.sub('<br(\s+)?/>(\s+)?', " ", detail)  # 去掉<br>
            data.append(detail.strip())  # strip()去掉空格
            data_list.append(data)
    return data_list


# 得到制定URL的网页内容
def askURL(url):
    head = {  # tell server what we are able to receive
        "User-Agent": "Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 86.0.4240.193 Safari / 537.36"
        # User Agent is a camouflage to tell the server that we are actually human

    }
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as result:
        print(result)
    return html


# 3 保存数据
def save_data(data_list, save_path):
    workbook = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
    worksheet = workbook.add_sheet('电影Top250', cell_overwrite_ok=True)  # 创建工作表
    col = ('电影详情链接', "图片链接", "中文名称", "外文名称", "港澳台名称", "评分", "评分人数", "一句介绍", "总览")
    for i in range(0, 9):
        worksheet.write(0, i, col[i])  # 列名
    for i in range(0, 250):
        print("第%d条" % (i+1))
        data = data_list[i]
        for g in range(0, 9):
            worksheet.write(i+1, g, data[g])  # Write Data
    workbook.save(save_path)  # Save


if __name__ == "__main__":
    main()
