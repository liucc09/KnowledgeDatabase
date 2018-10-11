# python读写txt
```python
with open("filename",'r+') as f:
    text = f.read()
    line = f.readline()
    lines = f.readlines()

    f.write("content")
    f.writelines(['hello dear!','hello son!','hello baby!'])
```
模式|描述
:---:|:---
r|读方式打开
w|写方式打开，如果存在则清空文件再写入
a|追加方式打开
r+|读写方式打开
w+|消除文件内容，以写方式打开
a+|读写方式打开，文件指针移到末尾
b|二进制方式打开