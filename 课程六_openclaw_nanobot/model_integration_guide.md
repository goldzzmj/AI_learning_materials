# OpenClaw 国产大模型集成指南

> 版本: 1.0  
> 更新日期: 2025年  
> 适用框架: OpenClaw AI Agent Framework

---

## 目录

1. [Kimi 模型集成](#1-kimi-模型集成)
2. [Qwen（通义千问）模型集成](#2-qwen通义千问-模型集成)
3. [其他国产模型](#3-其他国产模型)
4. [通用集成方法](#4-通用集成方法)
5. [最佳实践](#5-最佳实践)

---

## 1. Kimi 模型集成

### 1.1 获取 API Key

1. 访问 [Moonshot AI 开放平台](https://platform.moonshot.cn/console/api-keys)
2. 注册并登录账号
3. 进入控制台 → 创建 API Key
4. 选择 `default` 默认项目
5. 复制生成的 API Key（格式：`sk-xxxxxxxx`）

### 1.2 配置步骤

#### 环境变量配置

```bash
# Linux/macOS
export MOONSHOT_API_KEY="sk-your-api-key"

# Windows (CMD)
set MOONSHOT_API_KEY=sk-your-api-key

# Windows (PowerShell)
$env:MOONSHOT_API_KEY="sk-your-api-key"
```

#### OpenClaw 配置文件

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "moonshot/kimi-k2-0905-preview"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "moonshot": {
        "baseUrl": "https://api.moonshot.cn/v1",
        "apiKey": "${MOONSHOT_API_KEY}",
        "api": "openai-completions",
        "models": [
          {
            "id": "kimi-k2-0905-preview",
            "name": "Kimi K2",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 256000,
            "maxTokens": 32768
          },
          {
            "id": "kimi-k2-turbo-preview",
            "name": "Kimi K2 Turbo",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 256000,
            "maxTokens": 8192
          },
          {
            "id": "kimi-k2-thinking",
            "name": "Kimi K2 Thinking",
            "reasoning": true,
            "input": ["text"],
            "contextWindow": 256000,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

### 1.3 代码示例

#### Python 调用示例

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-your-moonshot-api-key",
    base_url="https://api.moonshot.cn/v1"
)

# 发送请求
completion = client.chat.completions.create(
    model="kimi-k2-0905-preview",
    messages=[
        {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ],
    temperature=0.6,
    max_tokens=32768
)

print(completion.choices[0].message.content)
```

#### Node.js 调用示例

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'sk-your-moonshot-api-key',
  baseURL: 'https://api.moonshot.cn/v1'
});

async function chat() {
  const completion = await client.chat.completions.create({
    model: 'kimi-k2-0905-preview',
    messages: [
      { role: 'system', content: '你是 Kimi，由 Moonshot AI 提供的人工智能助手。' },
      { role: 'user', content: '你好，请介绍一下自己。' }
    ],
    temperature: 0.6,
    max_tokens: 32768
  });
  
  console.log(completion.choices[0].message.content);
}

chat();
```

#### 流式输出示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-moonshot-api-key",
    base_url="https://api.moonshot.cn/v1"
)

stream = client.chat.completions.create(
    model="kimi-k2-0905-preview",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    temperature=0.6
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### 1.4 参数设置建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `temperature` | 0.6 | K2 模型推荐值，平衡创造性和准确性 |
| `max_tokens` | 32768 (K2) / 8192 (Turbo) | 根据模型选择 |
| `top_p` | 0.9 | 核采样参数 |
| `contextWindow` | 256000 | K2 系列支持 256K 上下文 |

### 1.5 可用模型列表

| 模型 ID | 上下文长度 | 特点 |
|---------|-----------|------|
| `kimi-k2-0905-preview` | 256K | 标准版，功能全面 |
| `kimi-k2-turbo-preview` | 256K | 快速版，响应更快 |
| `kimi-k2-thinking` | 256K | 思考模式，适合复杂推理 |
| `kimi-k2-thinking-turbo` | 256K | 快速思考模式 |

---

## 2. Qwen（通义千问）模型集成

### 2.1 获取 API Key

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 注册阿里云账号并完成实名认证
3. 进入「API Key 管理」页面
4. 创建新的 API Key（格式：`sk-xxxxxxxx`）

**注意**：北京地域和新加坡地域的 API Key 不同，请根据实际需求选择。

### 2.2 配置步骤

#### 环境变量配置

```bash
# Linux/macOS
export DASHSCOPE_API_KEY="sk-your-api-key"

# Windows (CMD)
set DASHSCOPE_API_KEY=sk-your-api-key

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="sk-your-api-key"
```

#### OpenClaw 配置文件

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "qwen/qwen-plus"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "qwen": {
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "${DASHSCOPE_API_KEY}",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen-plus",
            "name": "通义千问 Plus",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 131072,
            "maxTokens": 8192
          },
          {
            "id": "qwen-max",
            "name": "通义千问 Max",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192
          },
          {
            "id": "qwen-turbo",
            "name": "通义千问 Turbo",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 8192,
            "maxTokens": 4096
          }
        ]
      }
    }
  }
}
```

### 2.3 代码示例

#### Python 调用示例（OpenAI SDK）

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"}
    ]
)

print(completion.choices[0].message.content)
```

#### Python 调用示例（DashScope SDK）

```python
import os
from dashscope import Generation
import dashscope

# 设置 API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}
]

response = Generation.call(
    model="qwen-plus",
    messages=messages,
    result_format="message"
)

if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    print(f"错误: {response.code} - {response.message}")
```

#### Node.js 调用示例

```javascript
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY,
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
});

async function chat() {
    const completion = await client.chat.completions.create({
        model: "qwen-plus",
        messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "你是谁？" }
        ]
    });
    
    console.log(completion.choices[0].message.content);
}

chat();
```

#### 流式输出示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-dashscope-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

stream = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0]?.delta?.content
    if content:
        print(content, end="")
```

### 2.4 参数设置建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `temperature` | 0.7 | 通用场景推荐值 |
| `max_tokens` | 8192 (Plus/Max) / 4096 (Turbo) | 根据模型选择 |
| `top_p` | 0.9 | 核采样参数 |
| `result_format` | "message" | DashScope SDK 推荐 |

### 2.5 可用模型列表

| 模型 ID | 上下文长度 | 特点 |
|---------|-----------|------|
| `qwen-plus` | 128K | 平衡性能与成本 |
| `qwen-max` | 32K | 最强性能 |
| `qwen-turbo` | 8K | 快速响应，成本更低 |
| `qwen-max-latest` | 32K | 最新版本 |

---

## 3. 其他国产模型

### 3.1 文心一言（百度千帆）

#### 获取 API Key

1. 访问 [百度智能云控制台](https://console.bce.baidu.com/)
2. 注册并完成实名认证
3. 进入「千帆大模型平台」
4. 创建应用，获取 `API Key` 和 `Secret Key`

#### 调用示例

```python
import requests
import json

# 获取 access_token
API_KEY = 'your-api-key'
SECRET_KEY = 'your-secret-key'

def get_access_token():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}"
    response = requests.post(url)
    return response.json().get("access_token")

# 调用文心一言 API
access_token = get_access_token()
url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={access_token}"

payload = json.dumps({
    "messages": [
        {"role": "user", "content": "你好"}
    ],
    "temperature": 0.8,
    "response_format": "json_object"
})

headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, data=payload)
print(response.json())
```

#### 可用模型

| 模型名称 | 说明 |
|---------|------|
| `ERNIE-4.0-8K` | 文心一言 4.0 |
| `ERNIE-3.5-8K` | 文心一言 3.5 |
| `ERNIE-Speed-8K` | 高速版 |

### 3.2 智谱 AI（ChatGLM）

#### 获取 API Key

1. 访问 [智谱 AI 开放平台](https://open.bigmodel.cn/)
2. 注册账号
3. 进入「API Keys」页面创建密钥

#### 调用示例

```python
import requests

class GLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    def chat(self, messages, model="glm-4", temperature=0.7, max_tokens=1024):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()

# 使用示例
client = GLMClient(api_key="your-glm-api-key")
response = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    model="glm-4"
)
print(response['choices'][0]['message']['content'])
```

#### 可用模型

| 模型 ID | 说明 |
|---------|------|
| `glm-4` | 旗舰版 |
| `glm-4-flash` | 快速版 |
| `glm-4-0520` | 稳定版 |

### 3.3 讯飞星火

#### 获取 API Key

1. 访问 [讯飞开放平台](https://xinghuo.xfyun.cn/)
2. 注册并完成实名认证
3. 创建应用
4. 领取 Token 额度（Spark Lite 免费）

#### 调用示例

```python
import requests
import json

# 讯飞星火 API 调用
APP_ID = "your-app-id"
API_KEY = "your-api-key"
API_SECRET = "your-api-secret"

def chat_with_spark(messages):
    url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}:{API_SECRET}"
    }
    
    payload = {
        "model": "spark-lite",  # 或 spark-pro, spark-max
        "messages": messages
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# 使用示例
response = chat_with_spark([
    {"role": "user", "content": "你好"}
])
print(response)
```

#### 可用模型

| 模型 ID | 说明 | 价格 |
|---------|------|------|
| `spark-lite` | 轻量版 | 免费 |
| `spark-pro` | 专业版 | 按量付费 |
| `spark-max` | 旗舰版 | 按量付费 |

---

## 4. 通用集成方法

### 4.1 OpenAI 兼容接口的使用

所有国产大模型都提供了 OpenAI 兼容的 API 接口，可以使用统一的调用方式：

```python
from openai import OpenAI

def create_client(provider, api_key):
    """创建兼容 OpenAI 格式的客户端"""
    
    base_urls = {
        "moonshot": "https://api.moonshot.cn/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "glm": "https://open.bigmodel.cn/api/paas/v4",
    }
    
    return OpenAI(
        api_key=api_key,
        base_url=base_urls.get(provider)
    )

# 使用示例
client = create_client("moonshot", "sk-your-key")
```

### 4.2 配置文件修改

#### 统一配置文件模板

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "moonshot/kimi-k2-0905-preview",
        "fallback": "qwen/qwen-plus"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "moonshot": {
        "baseUrl": "https://api.moonshot.cn/v1",
        "apiKey": "${MOONSHOT_API_KEY}",
        "api": "openai-completions"
      },
      "qwen": {
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "${DASHSCOPE_API_KEY}",
        "api": "openai-completions"
      },
      "wenxin": {
        "baseUrl": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
        "apiKey": "${WENXIN_API_KEY}",
        "secretKey": "${WENXIN_SECRET_KEY}",
        "api": "baidu-wenxin"
      },
      "glm": {
        "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
        "apiKey": "${GLM_API_KEY}",
        "api": "openai-completions"
      }
    }
  }
}
```

### 4.3 环境变量设置

#### 创建 `.env` 文件

```bash
# Kimi / Moonshot
MOONSHOT_API_KEY=sk-your-moonshot-key

# 通义千问 / DashScope
DASHSCOPE_API_KEY=sk-your-dashscope-key

# 百度文心一言
WENXIN_API_KEY=your-wenxin-key
WENXIN_SECRET_KEY=your-wenxin-secret

# 智谱 AI
GLM_API_KEY=your-glm-key

# 讯飞星火
SPARK_APP_ID=your-spark-app-id
SPARK_API_KEY=your-spark-key
SPARK_API_SECRET=your-spark-secret
```

#### 加载环境变量

```python
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 获取 API Key
moonshot_key = os.getenv("MOONSHOT_API_KEY")
dashscope_key = os.getenv("DASHSCOPE_API_KEY")
```

---

## 5. 最佳实践

### 5.1 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 通用对话 | Kimi K2 / Qwen Plus | 平衡性能与成本 |
| 代码生成 | Kimi K2 / Qwen Max | 强大的代码能力 |
| 长文档处理 | Kimi K2 (256K) | 超长上下文支持 |
| 快速响应 | Qwen Turbo / Kimi Turbo | 低延迟 |
| 复杂推理 | Kimi K2 Thinking | 思考模式 |
| 成本敏感 | Spark Lite / Qwen Turbo | 免费或低价 |

### 5.2 成本控制

#### 价格对比（每百万 Token）

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| Kimi K2 | ¥10 | ¥30 |
| Qwen Plus | ¥2 | ¥6 |
| Qwen Max | ¥20 | ¥60 |
| GLM-4 | ¥0.8 | ¥2 |
| Spark Lite | 免费 | 免费 |

#### 成本优化策略

1. **选择合适的模型**：非复杂任务使用轻量级模型
2. **控制上下文长度**：避免发送不必要的历史消息
3. **设置合理的 max_tokens**：防止过度生成
4. **使用缓存**：对重复查询进行缓存
5. **批量处理**：合并多个请求

### 5.3 性能优化

#### 流式输出

```python
# 启用流式输出提升用户体验
stream = client.chat.completions.create(
    model="kimi-k2-0905-preview",
    messages=messages,
    stream=True  # 启用流式
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        # 实时输出到前端
        yield content
```

#### 连接池配置

```python
import requests
from requests.adapters import HTTPAdapter

# 创建带连接池的 session
session = requests.Session()
session.mount('https://', HTTPAdapter(pool_connections=10, pool_maxsize=20))

# 使用 session 发送请求
response = session.post(url, headers=headers, json=payload)
```

#### 重试机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_api_with_retry(client, messages):
    """带重试机制的 API 调用"""
    return client.chat.completions.create(
        model="kimi-k2-0905-preview",
        messages=messages
    )
```

### 5.4 错误处理

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

client = OpenAI(api_key="your-key", base_url="https://api.moonshot.cn/v1")

try:
    response = client.chat.completions.create(
        model="kimi-k2-0905-preview",
        messages=[{"role": "user", "content": "你好"}]
    )
except RateLimitError:
    # 请求频率超限
    print("请求过于频繁，请稍后重试")
except APIConnectionError:
    # 连接错误
    print("网络连接失败，请检查网络")
except APIError as e:
    # API 错误
    print(f"API 错误: {e.message}")
except Exception as e:
    # 其他错误
    print(f"未知错误: {str(e)}")
```

### 5.5 多模型降级策略

```python
class MultiModelClient:
    """多模型客户端，支持自动降级"""
    
    def __init__(self):
        self.clients = {
            "primary": OpenAI(api_key=os.getenv("MOONSHOT_API_KEY"), base_url="https://api.moonshot.cn/v1"),
            "secondary": OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "tertiary": OpenAI(api_key=os.getenv("GLM_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4")
        }
        self.models = {
            "primary": "kimi-k2-0905-preview",
            "secondary": "qwen-plus",
            "tertiary": "glm-4"
        }
    
    def chat(self, messages):
        """带降级机制的对话"""
        for level in ["primary", "secondary", "tertiary"]:
            try:
                client = self.clients[level]
                model = self.models[level]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=30
                )
                
                return {
                    "content": response.choices[0].message.content,
                    "model": model,
                    "level": level
                }
                
            except Exception as e:
                print(f"{level} 模型调用失败: {str(e)}")
                continue
        
        raise Exception("所有模型均调用失败")

# 使用示例
multi_client = MultiModelClient()
result = multi_client.chat([{"role": "user", "content": "你好"}])
print(f"使用模型: {result['model']}, 回复: {result['content']}")
```

---

## 附录

### A. 依赖安装

```bash
# Python
pip install openai>=1.0 python-dotenv requests tenacity

# Node.js
npm install openai dotenv axios
```

### B. 参考链接

- [Moonshot AI 开放平台](https://platform.moonshot.cn/)
- [阿里云百炼控制台](https://bailian.console.aliyun.com/)
- [百度智能云千帆](https://console.bce.baidu.com/qianfan/)
- [智谱 AI 开放平台](https://open.bigmodel.cn/)
- [讯飞星火开放平台](https://xinghuo.xfyun.cn/)

### C. 版本要求

| 组件 | 最低版本 |
|------|---------|
| Python | 3.8+ |
| Node.js | 18+ |
| OpenAI SDK | 1.0+ |

---

*文档结束*
