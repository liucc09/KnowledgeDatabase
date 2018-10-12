- [python](#python)
    - [python读写txt](#python%E8%AF%BB%E5%86%99txt)
- [jupyter](#jupyter)
    - [jupyter拓展组件](#jupyter%E6%8B%93%E5%B1%95%E7%BB%84%E4%BB%B6)
    - [jupyter调试](#jupyter%E8%B0%83%E8%AF%95)
# python
## python读写txt
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

# jupyter
## jupyter拓展组件
- `conda install -c conda-forge jupyter_contrib_nbextensions`
- `jupyter contrib nbextension install --user`

## jupyter调试
- 出错后使用`%debug`
- 在需要中断的地方加`import pdb,pdb.set_trace()`