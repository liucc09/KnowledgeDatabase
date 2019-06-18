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
- 查看远程仓库信息：`git remote -v`
# git 服务器
1. `git init --bare name.git` 建立远程仓库。服务器仓库必须是bare仓库，不然客户的无法 `push` 到 `checkout` 的分支上
2. 客户端 `git clone user@url:/path/name.git` 克隆远程仓库，或者 先`git init`再`git remote add origin user@url:/path/name.git` 添加远程仓库地址
---
# Reference
- [Git中文命令合集](https://www.yiibai.com/git)
- [Git官方文档](https://git-scm.com/book/zh/v1/%E8%B5%B7%E6%AD%A5)