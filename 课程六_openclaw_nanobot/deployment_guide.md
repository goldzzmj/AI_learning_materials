# OpenClaw & NanoBot 跨平台部署指南

> **版本**: 1.0  
> **更新日期**: 2025年  
> **适用范围**: Windows / Linux / macOS / Docker

---

## 目录

1. [项目概述](#1-项目概述)
2. [Windows部署指南](#2-windows部署指南)
3. [Linux部署指南](#3-linux部署指南)
4. [MacOS部署指南](#4-macos部署指南)
5. [Docker部署指南](#5-docker部署指南)
6. [常见问题排查](#6-常见问题排查)
7. [安全配置建议](#7-安全配置建议)

---

## 1. 项目概述

### 1.1 OpenClaw 简介

OpenClaw 是一个开源的个人AI助手，可以在您自己的硬件上运行。它连接到您常用的消息平台（WhatsApp、Telegram、Slack、Discord等），可以执行任务、管理日历、浏览网页、整理文件和运行终端命令。

**技术栈**:
- **语言**: TypeScript / Node.js
- **架构**: Gateway-Centric 架构
- **默认端口**: 
  - Gateway: 18789
  - Canvas Host: 18793
  - Browser Control: 18791

**官方资源**:
- GitHub: https://github.com/openclaw/openclaw
- 文档: https://docs.openclaw.ai

### 1.2 NanoBot 简介

NanoBot 是一个轻量级的MCP Agent构建工具，支持多种模型提供商和本地模型部署。

**技术栈**:
- **语言**: Go (nanobot-ai/nanobot) / Python (HKUDS/nanobot)
- **默认端口**:
  - 服务端口: 8080
  - UI开发端口: 5173

**官方资源**:
- Go版本: https://github.com/nanobot-ai/nanobot
- Python版本: https://github.com/HKUDS/nanobot

---

## 2. Windows部署指南

### 2.1 OpenClaw Windows部署

#### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10/11 (64位) | Windows 11 |
| 内存 | 4 GB RAM | 8 GB RAM |
| 存储空间 | 10 GB 可用空间 | 20 GB SSD |
| 网络 | 宽带互联网连接 | 高速连接 |
| WSL2 | 必需 | 推荐 |

#### 环境准备

**步骤1: 启用WSL2**

```powershell
# 以管理员身份运行PowerShell
wsl --install

# 设置WSL默认版本为2
wsl --set-default-version 2

# 安装Ubuntu
wsl --install -d Ubuntu-22.04
```

**步骤2: 安装Docker Desktop**

```powershell
# 下载并安装Docker Desktop
# 访问: https://www.docker.com/products/docker-desktop

# 验证安装
docker --version
docker-compose --version
```

**步骤3: 安装Node.js (可选，用于源码部署)**

```powershell
# 使用Chocolatey安装
choco install nodejs-lts

# 或使用nvm-windows
nvm install 20
nvm use 20

# 验证安装
node --version
npm --version
```

#### 详细安装步骤

**方法一: Docker部署（推荐）**

```powershell
# 1. 打开WSL Ubuntu终端
wsl -d Ubuntu-22.04

# 2. 创建OpenClaw目录
mkdir -p ~/.openclaw
cd ~/.openclaw

# 3. 创建docker-compose.yml文件
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"
      - "18793:18793"
    volumes:
      - ~/.clawdbot:/root/.clawdbot
      - ~/clawd:/root/clawd
    environment:
      - NODE_ENV=production
    command: openclaw gateway
EOF

# 4. 启动服务
docker-compose up -d

# 5. 查看日志
docker-compose logs -f
```

**方法二: 源码部署**

```powershell
# 1. 在WSL中克隆仓库
wsl -d Ubuntu-22.04
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 2. 安装依赖（使用pnpm推荐）
npm install -g pnpm
pnpm install
pnpm ui:build
pnpm build

# 3. 安装守护进程
openclaw onboard --install-daemon

# 4. 配置API密钥
openclaw config set anthropic.apiKey YOUR_API_KEY
openclaw config set openai.baseUrl https://api.openai.com/v1

# 5. 启动服务
openclaw start
```

#### 配置说明

**配置文件位置**: `~/.openclaw/.env`

```bash
# 编辑配置文件
nano ~/.openclaw/.env

# 添加以下内容
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 可选配置
CLAWDBOT_CONFIG_PATH=/root/.clawdbot/moltbot.json
CLAWDBOT_STATE_DIR=/root/.clawdbot
```

**访问控制面板**:
- 浏览器访问: `http://localhost:18789`
- 首次访问需要配置Gateway Token

---

### 2.2 NanoBot Windows部署

#### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10/11 | Windows 11 |
| Go版本 | 1.21+ | 1.22+ |
| Python版本 | 3.9+ (Python版本) | 3.11+ |
| 内存 | 2 GB RAM | 4 GB RAM |

#### Go版本部署

**步骤1: 安装Go**

```powershell
# 下载Go安装包
# 访问: https://go.dev/dl/

# 或使用Chocolatey
choco install golang

# 验证安装
go version
```

**步骤2: 安装NanoBot**

```powershell
# 1. 克隆仓库
git clone https://github.com/nanobot-ai/nanobot.git
cd nanobot

# 2. 构建项目
make

# 3. 运行NanoBot
./nanobot
```

**步骤3: 配置NanoBot**

```powershell
# 创建配置目录
mkdir -p ~/.nanobot

# 创建配置文件
cat > ~/.nanobot/config.json << 'EOF'
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
EOF
```

#### Python版本部署

```powershell
# 1. 安装Python（推荐3.11+）
# 访问: https://www.python.org/downloads/

# 2. 创建虚拟环境
python -m venv nanobot-env
.\nanobot-env\Scripts\activate

# 3. 安装NanoBot
pip install nanobot-ai

# 4. 初始化配置
nanobot init

# 5. 编辑配置文件
notepad %USERPROFILE%\.nanobot\config.json

# 6. 运行
nanobot agent -m "Hello World"
```

---

## 3. Linux部署指南

### 3.1 OpenClaw Linux部署

#### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Ubuntu 20.04+ / Debian 11+ | Ubuntu 22.04 LTS |
| 内存 | 4 GB RAM | 8 GB RAM |
| 存储空间 | 10 GB | 20 GB SSD |
| CPU | 2核 | 4核 |

#### 环境准备

**步骤1: 更新系统**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

**步骤2: 安装Docker**

```bash
# Ubuntu/Debian
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 启动Docker
sudo systemctl start docker
sudo systemctl enable docker

# 添加用户到docker组
sudo usermod -aG docker $USER
newgrp docker
```

**步骤3: 安装Node.js (可选)**

```bash
# 使用NodeSource
 curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# 或使用nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# 验证
node --version
npm --version
```

#### 详细安装步骤

**方法一: 一键安装脚本（推荐）**

```bash
# Linux/macOS一键安装
bash <(curl -fsSL https://raw.githubusercontent.com/phioranex/openclaw-docker/main/install.sh)

# 配置API密钥
nano ~/.openclaw/.env

# 启动服务
cd ~/.openclaw
docker compose up -d openclaw-gateway
```

**方法二: 手动Docker部署**

```bash
# 1. 创建目录
mkdir -p ~/.openclaw
cd ~/.openclaw

# 2. 创建docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"
      - "18793:18793"
    volumes:
      - ~/.clawdbot:/root/.clawdbot
      - ~/clawd:/root/clawd
    environment:
      - NODE_ENV=production
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    command: openclaw gateway
    networks:
      - openclaw-network

networks:
  openclaw-network:
    driver: bridge
EOF

# 3. 创建环境变量文件
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here
EOF

# 4. 启动服务
docker-compose up -d

# 5. 查看日志
docker-compose logs -f
```

**方法三: 源码部署**

```bash
# 1. 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 2. 安装pnpm
npm install -g pnpm

# 3. 安装依赖
pnpm install
pnpm ui:build
pnpm build

# 4. 安装守护进程
sudo ./deploy/deploy.sh
openclaw onboard --install-daemon

# 5. 配置API密钥
openclaw config set anthropic.apiKey YOUR_API_KEY

# 6. 启动服务
sudo systemctl start openclaw-gateway
sudo systemctl enable openclaw-gateway
```

#### 配置说明

**Systemd服务配置**:

```bash
# 创建systemd服务文件
sudo cat > /etc/systemd/system/openclaw-gateway.service << 'EOF'
[Unit]
Description=OpenClaw Gateway Service
After=network.target

[Service]
Type=simple
User=openclaw
WorkingDirectory=/home/openclaw/openclaw
ExecStart=/usr/bin/openclaw gateway
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

# 重新加载systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start openclaw-gateway
sudo systemctl enable openclaw-gateway

# 查看状态
sudo systemctl status openclaw-gateway
```

---

### 3.2 NanoBot Linux部署

#### Go版本部署

```bash
# 1. 安装Go
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# 2. 克隆并构建
git clone https://github.com/nanobot-ai/nanobot.git
cd nanobot
make

# 3. 安装到系统
sudo cp nanobot /usr/local/bin/

# 4. 初始化配置
nanobot init
```

#### Python版本部署

```bash
# 1. 安装Python和pip
sudo apt install -y python3 python3-pip python3-venv

# 2. 创建虚拟环境
python3 -m venv ~/.nanobot-env
source ~/.nanobot-env/bin/activate

# 3. 安装NanoBot
pip install nanobot-ai

# 4. 或使用uv安装
pip install uv
uv tool install nanobot-ai

# 5. 初始化
nanobot init

# 6. 配置
nano ~/.nanobot/config.json
```

---

## 4. MacOS部署指南

### 4.1 OpenClaw MacOS部署

#### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | macOS 12 (Monterey) | macOS 14 (Sonoma) |
| 内存 | 8 GB RAM | 16 GB RAM |
| 存储空间 | 10 GB | 20 GB SSD |
| 芯片 | Intel / Apple Silicon | Apple Silicon M2+ |

#### 环境准备

**步骤1: 安装Homebrew**

```bash
# 安装Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 添加到PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**步骤2: 安装Docker Desktop**

```bash
# 使用Homebrew安装
brew install --cask docker

# 启动Docker
open /Applications/Docker.app

# 验证
docker --version
```

**步骤3: 安装Node.js**

```bash
# 使用Homebrew安装
brew install node@20

# 或使用nvm
brew install nvm
source $(brew --prefix nvm)/nvm.sh
nvm install 20
nvm use 20
```

#### 详细安装步骤

**方法一: Docker部署（推荐）**

```bash
# 1. 创建目录
mkdir -p ~/.openclaw
cd ~/.openclaw

# 2. 创建docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    platform: linux/arm64  # Apple Silicon
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"
      - "18793:18793"
    volumes:
      - ~/.clawdbot:/root/.clawdbot
      - ~/clawd:/root/clawd
    environment:
      - NODE_ENV=production
    command: openclaw gateway
EOF

# 3. 启动
docker-compose up -d
```

**方法二: 一键安装脚本**

```bash
# 使用官方安装脚本
bash <(curl -fsSL https://openclaw.ai/install.sh)

# 配置API密钥
nano ~/.openclaw/.env

# 启动
cd ~/.openclaw
docker compose up -d
```

**方法三: 源码部署**

```bash
# 1. 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 2. 安装依赖
brew install pnpm
pnpm install
pnpm ui:build
pnpm build

# 3. 配置
openclaw config set anthropic.apiKey YOUR_API_KEY

# 4. 启动
openclaw start
```

#### 配置说明

**Mac Mini M4推荐配置**:

```bash
# 对于本地模型推理，推荐Mac Mini M4 16GB+
# 配置本地模型支持
openclaw config set localModel.enabled true
openclaw config set localModel.path /path/to/local/model
```

---

### 4.2 NanoBot MacOS部署

#### Go版本部署

```bash
# 1. 安装Go
brew install go

# 2. 克隆仓库
git clone https://github.com/nanobot-ai/nanobot.git
cd nanobot

# 3. 构建
make

# 4. 安装
sudo cp nanobot /usr/local/bin/

# 5. 初始化
nanobot init
```

#### Python版本部署

```bash
# 1. 安装Python
brew install python@3.11

# 2. 创建虚拟环境
python3 -m venv ~/.nanobot-env
source ~/.nanobot-env/bin/activate

# 3. 安装NanoBot
pip install nanobot-ai

# 4. 或使用uv
brew install uv
uv tool install nanobot-ai

# 5. 初始化配置
nanobot init

# 6. 编辑配置
nano ~/.nanobot/config.json
```

---

## 5. Docker部署指南

### 5.1 OpenClaw Docker部署

#### Docker Compose完整配置

```yaml
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"  # Gateway WebSocket
      - "18793:18793"  # Canvas Host
    volumes:
      - ~/.clawdbot:/root/.clawdbot    # 配置和凭证
      - ~/clawd:/root/clawd            # 工作空间
    environment:
      - NODE_ENV=production
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}
      - CLAWDBOT_CONFIG_PATH=/root/.clawdbot/moltbot.json
      - CLAWDBOT_STATE_DIR=/root/.clawdbot
    command: openclaw gateway
    networks:
      - openclaw-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:18789/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  openclaw-network:
    driver: bridge
```

#### 部署步骤

```bash
# 1. 创建工作目录
mkdir -p ~/openclaw-docker
cd ~/openclaw-docker

# 2. 创建docker-compose.yml（如上）

# 3. 创建环境变量文件
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-api03-xxx
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
EOF

# 4. 创建数据目录
mkdir -p ~/.clawdbot ~/clawd

# 5. 启动服务
docker-compose up -d

# 6. 查看日志
docker-compose logs -f

# 7. 访问控制面板
# http://localhost:18789
```

#### 常用Docker命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 更新镜像
docker-compose pull
docker-compose up -d

# 进入容器
docker exec -it openclaw /bin/bash

# 查看容器状态
docker ps

# 备份数据
docker run --rm -v openclaw_clawdbot:/data -v $(pwd):/backup alpine tar czf /backup/clawdbot-backup.tar.gz -C /data .
```

### 5.2 NanoBot Docker部署

```dockerfile
# Dockerfile for NanoBot (Python版本)
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install nanobot-ai

# 创建配置目录
RUN mkdir -p /root/.nanobot

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["nanobot", "agent"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  nanobot:
    build: .
    container_name: nanobot
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ~/.nanobot:/root/.nanobot
    environment:
      - NANOBOT_CONFIG_PATH=/root/.nanobot/config.json
```

---

## 6. 常见问题排查

### 6.1 OpenClaw常见问题

#### 问题1: 端口被占用

**症状**:
```
Error: listen EADDRINUSE: address already in use :::18789
```

**解决方案**:
```bash
# 查找占用端口的进程
# Linux/macOS
lsof -i :18789

# Windows
netstat -ano | findstr :18789

# 终止进程
# Linux/macOS
kill -9 <PID>

# Windows
taskkill /PID <PID> /F

# 或修改OpenClaw端口
openclaw config set gateway.port 18889
```

#### 问题2: Docker容器无法启动

**症状**:
```
Error: container openclaw is restarting
```

**解决方案**:
```bash
# 查看详细日志
docker logs openclaw

# 检查权限
sudo chown -R $USER:$USER ~/.clawdbot ~/clawd

# 重新创建容器
docker-compose down
docker-compose up -d
```

#### 问题3: API密钥配置错误

**症状**:
```
Error: Invalid API key provided
```

**解决方案**:
```bash
# 检查配置文件
cat ~/.openclaw/.env

# 重新配置
openclaw config set anthropic.apiKey YOUR_NEW_API_KEY

# 重启服务
docker-compose restart
```

#### 问题4: 内存不足

**症状**:
```
Error: JavaScript heap out of memory
```

**解决方案**:
```bash
# 增加Node.js内存限制
export NODE_OPTIONS="--max-old-space-size=4096"

# Docker部署时修改docker-compose.yml
environment:
  - NODE_OPTIONS=--max-old-space-size=4096
```

#### 问题5: WSL2网络问题（Windows）

**症状**:
无法从Windows主机访问WSL2中的OpenClaw服务

**解决方案**:
```powershell
# 在WSL2中获取IP地址
ip addr show eth0 | grep 'inet\b' | awk '{print $2}' | cut -d/ -f1

# 或使用localhost转发
# 在Windows PowerShell中运行
netsh interface portproxy add v4tov4 listenport=18789 connectport=18789 connectaddress=<WSL2_IP>

# 或者使用WSL2的localhost转发功能
# 在.wslconfig中配置
```

### 6.2 NanoBot常见问题

#### 问题1: 模型连接失败

**症状**:
```
Error: Connection refused to model API
```

**解决方案**:
```bash
# 检查配置文件
cat ~/.nanobot/config.json

# 验证API密钥
# 测试OpenRouter连接
curl -H "Authorization: Bearer YOUR_API_KEY" https://openrouter.ai/api/v1/models

# 检查网络连接
ping openrouter.ai
```

#### 问题2: 本地模型无法连接（vLLM）

**症状**:
```
Error: Failed to connect to vLLM server
```

**解决方案**:
```bash
# 启动vLLM服务器
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 检查服务器状态
curl http://localhost:8000/v1/models

# 更新配置
nano ~/.nanobot/config.json
```

#### 问题3: Go版本构建失败

**症状**:
```
Error: make: *** No targets specified
```

**解决方案**:
```bash
# 检查Go版本
go version

# 安装依赖
go mod download

# 手动构建
go build -o nanobot .
```

---

## 7. 安全配置建议

### 7.1 生产环境安全清单

| 安全层级 | 开发环境 | 生产环境 |
|----------|----------|----------|
| Gateway绑定 | 0.0.0.0 | 127.0.0.1 |
| mDNS | 启用 | 禁用 |
| 控制UI | 启用 | 禁用或认证 |
| 认证Token | 可选 | 必需 |
| 凭证存储 | 明文 | 加密 |
| 审计日志 | 可选 | 必需 |
| 出口控制 | 无 | 白名单 |
| 审批流程 | 无 | 高风险操作 |
| 容器隔离 | 可选 | 必需 |
| 监控 | 可选 | 必需 |

### 7.2 网络安全配置

```bash
# 1. 绑定到本地回环
openclaw config set gateway.bind 127.0.0.1

# 2. 禁用mDNS
openclaw config set mdns.enabled false

# 3. 启用Token认证
openclaw config set auth.token YOUR_SECURE_TOKEN

# 4. 配置防火墙
# Ubuntu/Debian
sudo ufw allow 18789/tcp
sudo ufw enable

# 或使用iptables
sudo iptables -A INPUT -p tcp --dport 18789 -s 127.0.0.1 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 18789 -j DROP
```

### 7.3 凭证安全

```bash
# 使用环境变量而非配置文件
export ANTHROPIC_API_KEY=your_key_here

# 或使用Docker Secrets
docker secret create anthropic_api_key <(echo "your_key")

# 定期轮换密钥
# 设置提醒每90天轮换一次
```

### 7.4 反向代理配置（Nginx）

```nginx
# /etc/nginx/sites-available/openclaw
server {
    listen 80;
    server_name your-domain.com;
    
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:18789;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 7.5 监控和日志

```bash
# 配置日志轮转
sudo cat > /etc/logrotate.d/openclaw << 'EOF'
/var/log/openclaw/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 openclaw openclaw
}
EOF

# 使用systemd日志
sudo journalctl -u openclaw-gateway -f

# 或使用Docker日志
docker logs -f openclaw --tail 100
```

---

## 附录

### A. 快速参考命令

```bash
# OpenClaw
openclaw start                    # 启动服务
openclaw stop                     # 停止服务
openclaw config set <key> <val>   # 设置配置
openclaw config get <key>         # 获取配置
openclaw logs                     # 查看日志

# NanoBot
nanobot init                      # 初始化
nanobot agent -m "message"        # 发送消息
nanobot agent --interactive       # 交互模式
```

### B. 端口参考

| 服务 | 端口 | 说明 |
|------|------|------|
| OpenClaw Gateway | 18789 | WebSocket服务器 |
| OpenClaw Canvas | 18793 | Canvas主机 |
| OpenClaw Browser | 18791 | 浏览器控制 |
| NanoBot Service | 8080 | 服务端口 |
| NanoBot UI Dev | 5173 | UI开发端口 |

### C. 官方资源链接

- OpenClaw GitHub: https://github.com/openclaw/openclaw
- OpenClaw 文档: https://docs.openclaw.ai
- NanoBot (Go): https://github.com/nanobot-ai/nanobot
- NanoBot (Python): https://github.com/HKUDS/nanobot

---

*文档版本: 1.0 | 最后更新: 2025年*
