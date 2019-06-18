- [单独线程后台运行程序且不产生输出文件](#%E5%8D%95%E7%8B%AC%E7%BA%BF%E7%A8%8B%E5%90%8E%E5%8F%B0%E8%BF%90%E8%A1%8C%E7%A8%8B%E5%BA%8F%E4%B8%94%E4%B8%8D%E4%BA%A7%E7%94%9F%E8%BE%93%E5%87%BA%E6%96%87%E4%BB%B6)
- [ssh传输文件](#ssh%E4%BC%A0%E8%BE%93%E6%96%87%E4%BB%B6)
    - [从服务器下载文件](#%E4%BB%8E%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8B%E8%BD%BD%E6%96%87%E4%BB%B6)
    - [上传本地文件到服务器](#%E4%B8%8A%E4%BC%A0%E6%9C%AC%E5%9C%B0%E6%96%87%E4%BB%B6%E5%88%B0%E6%9C%8D%E5%8A%A1%E5%99%A8)
    - [操作整个目录](#%E6%93%8D%E4%BD%9C%E6%95%B4%E4%B8%AA%E7%9B%AE%E5%BD%95)
- [删除文件夹](#%E5%88%A0%E9%99%A4%E6%96%87%E4%BB%B6%E5%A4%B9)
- [shell script](#shell-script)
- [解压](#%E8%A7%A3%E5%8E%8B)
- [解压中文乱码](#%E8%A7%A3%E5%8E%8B%E4%B8%AD%E6%96%87%E4%B9%B1%E7%A0%81)

# 单独线程后台运行程序且不产生输出文件
```bash
nohup java -jar /xxx/xxx/xxx.jar >/dev/null 2>&1 &
```
关键在于最后的 `>/dev/null 2>&1` 部分，`/dev/null` 是一个虚拟的空设备（类似物理中的黑洞），任何输出信息被重定向到该设备后，将会石沉大海

`>/dev/null` 表示将标准输出信息重定向到"黑洞"

`2>&1` 表示将标准错误重定向到标准输出(由于标准输出已经定向到“黑洞”了，即：标准输出此时也是"黑洞"，再将标准错误输出定向到标准输出，相当于错误输出也被定向至“黑洞”)

# ssh传输文件
## 从服务器下载文件
`scp username@servername:/path/filename /var/www/local_dir（本地目录）`
>例如 `scp root@192.168.0.101:/var/www/test.txt /var/www/local_dir` 把 `192.168.0.101` 上的`/var/www/test.txt` 的文件下载到 `/var/www/local_dir`（本地目录）
## 上传本地文件到服务器
`scp /path/filename username@servername:/path`
>例如 `scp /var/www/test.php  root@192.168.0.101:/var/www/` 把本机 `/var/www/` 目录下的 `test.php` 文件上传到 `192.168.0.101` 这台服务器上的 `/var/www/` 目录中
## 操作整个目录
`scp -r username@servername:/var/www/remote_dir/（远程目录） /var/www/local_dir（本地目录）`

# 删除文件夹
- 命令：`rm`
- -r 就是向下递归，不管有多少级目录，一并删除
- -f 就是直接强行删除，不作任何提示的意思

实例：
`rm -rf /opt/svn` 将会删除/opt/svn/目录以及其下所有文件夹，包括文件

# shell script
```bash
#!/bin/bash
nohup jupyter notebook>/dev/null 2>&1 &
```
# 解压
- `tar.gz` : `tar -zxvf ×××.tar.gz`
- `tar.bz2` : `tar -jxvf ×××.tar.bz2`

# 解压中文乱码
- `unzip -O cp936 file.zip`

#安装deb
```bash
sudo dpkg -i name.deb
```

#查看进程
```bash
ps -aux
```