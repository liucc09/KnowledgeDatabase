- [python](#python)
  - [python读写txt](#python%E8%AF%BB%E5%86%99txt)
- [jupyter](#jupyter)
  - [jupyter拓展组件](#jupyter%E6%8B%93%E5%B1%95%E7%BB%84%E4%BB%B6)
  - [查看本地变量](#%E6%9F%A5%E7%9C%8B%E6%9C%AC%E5%9C%B0%E5%8F%98%E9%87%8F)
  - [重新加载模块](#%E9%87%8D%E6%96%B0%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9D%97)
  - [后台运行jupyter notebook](#%E5%90%8E%E5%8F%B0%E8%BF%90%E8%A1%8Cjupyter-notebook)
  - [jupyter notebook 调试技巧](#jupyter-notebook-%E8%B0%83%E8%AF%95%E6%8A%80%E5%B7%A7)
  - [jupyter notebook 中执行 shell 命令](#jupyter-notebook-%E4%B8%AD%E6%89%A7%E8%A1%8C-shell-%E5%91%BD%E4%BB%A4)
  - [jupyter shell 中访问变量](#jupyter-shell-%E4%B8%AD%E8%AE%BF%E9%97%AE%E5%8F%98%E9%87%8F)
  - [jupyter 中使用tqdm进度条](#jupyter-%E4%B8%AD%E4%BD%BF%E7%94%A8tqdm%E8%BF%9B%E5%BA%A6%E6%9D%A1)
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
- 安装pixiedust`pip install pixiedust`
  `import pixiedust`
  在单元格上方输入`%%pixie_debugger`
  `%%pixie_debugger -b find_max 9`设置断点位置
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

## jupyter 分析程序运行时间
- 使用prun
```python
%prun sum_of_list(10000)
```
- 使用line_profiler逐行分析
```python
%load_ext line_profiler
%lprun -f sum_of_lists sum_of_lists(5000)
```