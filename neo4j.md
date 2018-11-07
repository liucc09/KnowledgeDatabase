# neo4j 入门教程

## ubuntu安装neo4j

- 安装java：`sudo apt install default-jre default-jre-headless`
- 添加仓库key：`wget --no-check-certificate -O - https://debian.neo4j.org/neotechnology.gpg.key | sudo apt-key add -`
- 添加仓库：`echo 'deb http://debian.neo4j.org/repo stable/' | sudo tee /etc/apt/sources.list.d/neo4j.list`
- 安装：
  
```bash
sudo apt update
sudo apt install neo4j
```

- 关闭启动服务：
  
```bash
sudo service neo4j stop
sudo service neo4j start
```

- 浏览器访问：<http://localhost:7474/browser/>

## 远程访问

- 在安装目录的 `$NEO4J_HOME/conf/neo4j.conf` 文件内，找到下面一行，将注释#号去掉就可以了 `dbms.connectors.default_listen_address=0.0.0.0`
- 访问链接：<http://localhost:7474/browser/>
- 用户名/密码：`neo4j/neo4j`