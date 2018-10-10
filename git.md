# 常用命令
- 克隆远程仓库：`git clone https://github.com/liucc09/KnowledgeDatabase.git`
- 添加track：`git add -A`
- 创建commit：`git commit -m "comments"`
- 推送到远程仓库(master->origin)：`git push origin master`
- 获取远程分支并合并：`git pull <远程主机名> <远程分支名>:<本地分支名>`
- 配置用户信息：
```git
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```
- git记住用户名和密码：`git config --global credential.helper store`   
- 忽略一些文件：使用`.gitignore`文件
- 设置别名：`git config --global alias.别名 '指定代码'`
---
# Reference
[Git中文命令合集](https://www.yiibai.com/git)