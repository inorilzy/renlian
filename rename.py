import os

path = input('请输入文件路径(结尾加上/)：')

# 获取该目录下所有文件，存入列表中
f = os.listdir(path)

n = 0
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + f[n]

    # 设置新文件名
    newname = path + 'a' + str(n + 1) + '.jpg'

    # 用os模块中的rename方法对文件改名
    os.rename(oldname, newname)
    print(oldname, '======>', newname)

    n += 1