- [subplots](#subplots)
- [坐标轴等比例](#坐标轴等比例)
# subplots
```python
fig, axs = plt.subplots(ncols=len(im_paths), sharex=True, sharey=True,figsize=(10,10))
for im,ax in zip(ims,axs):
    ime = feature.canny(im)
    ax.imshow(ime, cmap=plt.cm.gray)
    ax.axis('off')
    
plt.tight_layout()    
plt.show()
```
# 坐标轴等比例
`plt.axis('equal')`