- [python](#python)
    - [python读写txt](#python读写txt)
- [jupyter](#jupyter)
    - [jupyter拓展组件](#jupyter拓展组件)
    - [查看本地变量](#查看本地变量)
    - [重新加载模块](#重新加载模块)
    - [后台运行jupyter notebook](#后台运行jupyter-notebook)
    - [jupyter notebook 调试技巧](#jupyter-notebook-调试技巧)
    - [jupyter notebook 中执行 shell 命令](#jupyter-notebook-中执行-shell-命令)
    - [jupyter shell 中访问变量](#jupyter-shell-中访问变量)
    - [jupyter 中使用tqdm进度条](#jupyter-中使用tqdm进度条)
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

## 查看本地变量
`%whos`
```python
Variable                  Type                 Data/Info
--------------------------------------------------------
np                        module               <module 'numpy' from '/ho<...>kages/numpy/__init__.py'>
os                        module               <module 'os' from '/home/<...>da3/lib/python3.6/os.py'>
pd                        module               <module 'pandas' from '/h<...>ages/pandas/__init__.py'>
plt                       module               <module 'matplotlib.pyplo<...>es/matplotlib/pyplot.py'>
re                        module               <module 're' from '/home/<...>da3/lib/python3.6/re.py'>
time                      module               <module 'time' (built-in)>
unittest                  module               <module 'unittest' from '<...>.6/unittest/__init__.py'>
url                       str                  http://data.house.163.com<...>districtname=全市#stoppoint
webdriver                 module               <module 'selenium.webdriv<...>m/webdriver/__init__.py'>
x                         ndarray              8: 8 elems, type `int64`, 64 bytes
y                         list                 n=57
```
## 重新加载模块
有时不小心把模块中的变量给赋值了，就需要重新加载
```python
import importlib
importlib.reload(plt)
```
## 后台运行jupyter notebook
有时不希望关闭shell时一同关闭jupyter notebook，则用下面命令打开
```bash
nohup jupyter notebook &
```
## jupyter notebook 调试技巧
- 在需要中断的地方输入`import pdb; pdb.set_trace()`，使用如下命令进行调试

命令 | 解释
-------------|--------------
break 或 b | 设置断点设置断点
continue 或 c | 继续执行程序,运行到下一个断点
list 或 l | 查看当前行的代码段 ，显示断点周围的源代码
step 或 s | 进入函数，步进，一步步的执行
return 或 r | 执行代码直到从当前函数返回
exit 或 q | 中止并退出
next 或 n | 执行下一行
pp | 打印变量的值help帮助
- 在出错后输入`%debug`可以查看出错时上下文的变量值
## jupyter notebook 中执行 shell 命令
- 使用 `!` 可以在单独进程中执行命令，不会对上下文产生影响，如`!cd ~`
- 使用 `%` 可以影响上下文
- 使用 `%%bash` 或 `%%cmd` 可以将整个单元格都转为用指定程序执行
```
%%bash
cd ~
ls -al
```
## jupyter shell 中访问变量
`$variable`

## jupyter 中使用tqdm进度条
`from tqdm import tqdm_notebook as tqdm`