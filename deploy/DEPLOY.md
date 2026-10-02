# aml-memory-adapter 部署文档（阿里云 ECS）

> 目标：把评测 API 部署到 `https://aml.ningxia-tour.top`，与反诈项目（fraud-api）同机隔离共存。
> 环境：Alibaba Cloud Linux 3（dnf 系）+ Nginx + MySQL + systemd + SELinux enforcing。
> 原则：独立进程 / 独立数据库 schema / 独立 .env / 独立子域名，不动反诈项目任何配置。
> 时间要求：部署完成后须保活至 **2026-11-04 23:59**。

---

## 0. 阿里云控制台（网页操作，约 10 分钟）

1. **DNS**：云解析 DNS → `ningxia-tour.top` → 添加记录：`A` 记录，主机记录 `aml`，值 `39.99.32.108`。
2. **证书**：数字证书管理服务 → 免费证书 → 申请 `aml.ningxia-tour.top`（单域名，自动带 `www.`，审核几分钟）→ 通过后下载 **Nginx 格式**（得到 `.pem` + `.key` 两个文件）。
3. 安全组：80/443 已开（反诈项目在用），**不需要**开 8600——服务只监听 127.0.0.1，全走 Nginx 反代。

## 1. 拉取代码

```bash
cd /opt
git clone https://github.com/w6403/aml-memory-adapter.git
# 若阿里云连 github 超时，用镜像：
# git clone https://ghfast.top/https://github.com/w6403/aml-memory-adapter.git
cd aml-memory-adapter
chown -R nginx:nginx /opt/aml-memory-adapter
```

公开仓库只读拉取，无需 deploy key。服务器上只做 `git pull`，不做 commit/push。

## 2. Python 环境（复用已装的 python3.11）

```bash
cd /opt/aml-memory-adapter
python3.11 -m venv .venv
.venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

## 3. MySQL 建库（独立 schema + 独立账号）

```bash
mysql -uroot -p
```

```sql
CREATE DATABASE aml_eval CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'aml_user'@'localhost' IDENTIFIED BY '【换一个强随机密码】';
GRANT ALL PRIVILEGES ON aml_eval.* TO 'aml_user'@'localhost';
FLUSH PRIVILEGES;
```

## 4. 配置 .env

```bash
cd /opt/aml-memory-adapter
cp .env.example .env
vi .env
```

```ini
DATABASE_URI=mysql+pymysql://aml_user:【上面设的密码】@localhost:3306/aml_eval
DASHSCOPE_API_KEY=【独立申请的 dashscope Key；无则留空，自动降级关键词模式】
EMBEDDING_MODEL=text-embedding-v3
API_AUTH_TOKEN=【API_AUTH_TOKEN：服务器 .env 里的值】

PORT=8600
```

> API_AUTH_TOKEN 建议设上：接口公网暴露，评测申请时把它作为 Memory System Key 提交给平台。

## 5. systemd 服务

```bash
cp /opt/aml-memory-adapter/deploy/aml-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now aml-api
systemctl status aml-api        # active (running) 即成功
journalctl -u aml-api -n 50     # 看日志
curl http://127.0.0.1:8600/health
# 期望: {"status":"ok","service":"aml-memory-adapter",...}
```

## 6. Nginx + 证书

```bash
# 证书文件（控制台下载的两个文件，从本地 scp 上传后）：
mkdir -p /etc/nginx/ssl
mv /root/aml.ningxia-tour.top.pem /root/aml.ningxia-tour.top.key /etc/nginx/ssl/

chmod 600 /etc/nginx/ssl/aml.ningxia-tour.top.key

cp /opt/aml-memory-adapter/deploy/nginx-aml.conf /etc/nginx/conf.d/aml-ningxia-tour.top.conf
nginx -t && systemctl reload nginx
```

## 7. SELinux 确认

反诈项目已放行过（全局布尔值），确认一下即可：

```bash
getsebool httpd_can_network_connect   # 期望 on；若不是：setsebool -P httpd_can_network_connect 1
```

## 8. 端到端验证

```bash
curl https://aml.ningxia-tour.top/health

# Add 写入（2026-10-02 起为官方契约格式：request_id / role / 响应 echo）
curl -X POST https://aml.ningxia-tour.top/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 【API_AUTH_TOKEN：服务器 .env 里的值】" \
  -d '{"request_id":"smoke:chunk-0","user_id":"smoke_u1","session_id":"s1","messages":[{"role":"user","content":"我是软件工程专业的大一新生"}]}'

# Search 检索（响应为 {data:[{id,content,score,created_at}]}）
curl -X POST https://aml.ningxia-tour.top/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 【API_AUTH_TOKEN：服务器 .env 里的值】" \
  -d '{"user_id":"smoke_u1","query":"用户的专业","top_k":5}'
```

三条都正常返回 = 部署完成，可以去官网提交评测申请（填 `https://aml.ningxia-tour.top/add` 和 `/search`）。

## 9. 日常更新（三步）

```bash
cd /opt/aml-memory-adapter && git pull
# 若 requirements.txt 有变化（如 2026-10-02 新增 numpy）：
.venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
systemctl restart aml-api
curl https://aml.ningxia-tour.top/health
# health 里 vector_weight 显示 0.5 即为 2026-10-02 后的版本
```

⚠️ **Full 评测申报后冻结代码**：git pull 只允许修 bug，不得改 API 契约/鉴权/字段。

## 10. 排错

| 现象 | 排查 |
|------|------|
| `systemctl status aml-api` failed | `journalctl -u aml-api -n 100` 看 traceback；多为 .env 的 DATABASE_URI 密码/库名错 |
| curl 502 | 服务没起或端口不对：`ss -tlnp \| grep 8600` |
| curl 404 | Nginx server_name 没匹配上：`nginx -T \| grep -A5 aml` |
| 证书报错 | 证书域名与访问域名不一致；确认下载的是 `aml.ningxia-tour.top` 的证书 |
| git pull 超时 | 换 ghfast.top 镜像：`git remote set-url origin https://ghfast.top/https://github.com/w6403/aml-memory-adapter.git` |
| embedding 不生效 | health 里 `embedding_available:false` → DASHSCOPE_API_KEY 没配或额度欠费，系统自动降级关键词模式，不影响接口可用 |
