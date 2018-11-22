- [subplots](#subplots)
- [坐标轴等比例](#%E5%9D%90%E6%A0%87%E8%BD%B4%E7%AD%89%E6%AF%94%E4%BE%8B)
- [中文字体乱码问题](#%E4%B8%AD%E6%96%87%E5%AD%97%E4%BD%93%E4%B9%B1%E7%A0%81%E9%97%AE%E9%A2%98)
    - [临时解决方法](#%E4%B8%B4%E6%97%B6%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95)
    - [一劳永逸解决方法](#%E4%B8%80%E5%8A%B3%E6%B0%B8%E9%80%B8%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95)
# subplots
```python
fig, axs = plt.subplots(ncols=len(im_paths), sharex=True, sharey=True,figsize=(10,10))
for im,ax in zip(ims,axs):
    ime = feature.canny(im)
    ax.imshow(ime, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_xlabel('x')

plt.tight_layout()    
plt.show()
```
# 坐标轴等比例
`plt.axis('equal')`

# 中文字体乱码问题
## 临时解决方法
- 下载SimHei字体
- 将字体放到`/home/liucc/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf`目录下
- 在调用matplotlib前调用代码
```python
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
```
## 一劳永逸解决方法
- 下载SimHei字体
- 将字体放到`/home/liucc/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf`目录下
- 打开`/home/liucc/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc`文件
- 取消注释 `font.family : sans-serif`
- 在最前面添加 `font.sans-serif : SimHei, ...`