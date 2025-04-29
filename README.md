<p align="center">
  <a href="https://github.com/lopinx/crawl4geo" target="_blank"><img src="https://cdn.lightpanda.io/assets/images/logo/lpd-logo.png" alt="Logo" height=170></a>
</p>

<h1 align="center">微信公众号助手</h1>

<p align="center"><a href="https://github.com/lopinx/WeChatMPCopilot">微信公众号助手</a></p>

<div align="center">

[![KeyBERT](https://img.shields.io/github/stars/MaartenGr/KeyBERT)](https://github.com/MaartenGr/KeyBERT)
[![jieba](https://img.shields.io/github/stars/fxsjy/jieba)](https://github.com/fxsjy/jieba)
[![NLTK](https://img.shields.io/github/stars/nltk/nltk)](https://github.com/nltk/nltk)

</div>

---

## 项目概述

`WeChatMPCopilot` 是一个自动化工具，旨在帮助用户通过微信公众平台 API 自动发布文章。该工具支持从关键词或标题生成文章内容，并通过 AI 模型（如 GPT）生成高质量的文章。此外，它还集成了图片上传、草稿保存和文章发布的完整流程。

主要功能包括：

- **关键词提取**：基于 KeyBERT 和 TextRank 算法提取文章关键词。

- **AI 内容生成**：通过 OpenAI 或其他大模型生成文章标题和内容。

- **微信公众号集成**：自动上传图片、保存草稿并发布文章。

- **多语言支持**：支持中文和英文内容生成。

- **SEO 优化**：生成的内容符合 SEO 最佳实践，提升文章曝光率。

---

## 安装依赖

### 使用 `uv` 工具安装依赖

1. 安装 `uv` 工具（如果尚未安装）：

   ```bash
   pip install uv
   ```

2. 使用 `uv` 添加依赖包：

   ```bash
   uv add httpx[http2,http3] keybert scikit-learn jieba nltk fuzzywuzzy python-Levenshtein markdown pygments pymdown-extensions markdownify openai pandas
   ```

3. 或者使用 `requirements.txt` 安装依赖：

   ```bash
   uv pip install -r requirements.txt
   ```

4. 导出依赖包：

   ```bash
   uv pip freeze | uv pip compile - -o requirements.txt
   ```

### 手动安装依赖

你也可以手动安装所有依赖包：

```bash
pip install httpx[http2,http3] keybert scikit-learn jieba nltk fuzzywuzzy python-Levenshtein markdown pygments pymdown-extensions markdownify openai pandas
```

---

## 配置文件说明

### `config.json`

`config.json` 文件包含项目的全局配置信息，包括微信公众号 API 的凭据、GPT 模型的配置以及文章生成的相关参数。以下是一个示例配置：

``` json5
{
    "gpts": [
      {
        "baseurl": "http://127.0.0.1:11434/v1", //API接口地址，/chat/completions 前部分
        "apikey": [
          "EMPTY"
        ],
        "models": [
          "deepseek-r1:latest"                  // GPT模型,列表引入，可以多个
        ]
      }
    ],
    "stopk": {
      "cn": "ChineseStopWords.txt",             // 停用词文件路径
      "en": "EnglishStopWords.txt"              // 停用词文件路径
    },
    "mps": [
      {
        "name": "吴罗平",                       // 公众号名称
        "appid": "",                            // 公众号 AppID
        "secret": "",                           // 公众号 AppSecret
        "baseurl": "https://api.weixin.qq.com", // 微信公众号 API 接口地址
        "lang": "中文",                          // 文章语言类型
        "aitags": true,                         // 是否使用GPT生成标签
        "aiexcerpt": true,                      // 是否使用GPT生成摘要
        "draft": false,                         // 是否上传草稿
        "publish": false,                       // 是否发布文章
        "keys": "keywords.txt",                  // 关键词文件路径
        // reqkeys: 核心关键词列表（影响分词结果，不是必须）
        "reqkeys": [
          "SEO",
          "网站建设",
          "网络营销",
          "网站优化",
          "网站推广",
          "网站流量",
          "网站运营",
          "微信公众号"
        ],
        "number": 3,                            // 每次最多生成文章数量
        "source_url": "",                         // 阅读原文 链接
        "author": "lopins",                      // 文章作者
        // 图片素材，从微信公众号素材库上传的图片素材ID（需要自己获取，本项目不提供：因为不是所有人都需要，关键词：素材ID）
        "media_ids": {
          "睡眠障碍（失眠）": "noV6uipu7YVLwbb16HbFYSJcwovsmrtAZjWhdKD_C1iCmv93lLTq6R7SdX3nOGN4",
          "尿毒症": "noV6uipu7YVLwbb16HbFYYhQUy0at2Acn1ht3318BxHkwLmRrbuQuDu8Lzr59ERI",
          "夫妻生活（两性）": "noV6uipu7YVLwbb16HbFYQVpJtBLVwJKybqkKFpo96Os5RohFpI5rCPeKELn_o1S",
          "糖尿病（胰岛素）": "noV6uipu7YVLwbb16HbFYe9djnLaHk18OCw4cknHNsJ0VbRar092a4bIPwUc0f7w",
          "美容抗衰 （国内）": "noV6uipu7YVLwbb16HbFYQzLytsMtfsdzQyCjZpNcVEs0WiB7Lo1v32QdzgbMnKc",
          "关节炎（膝关节）": "noV6uipu7YVLwbb16HbFYey40BtXdboB-AMCY676hm03DHYL75xQZRgkoyUG05sZ",
          "肝硬化": "noV6uipu7YVLwbb16HbFYT7zdck2-QW976rR7GOf8SrdtKgrmYYA9sQXoxoAdSvJ",
          "肺（慢阻肺）": "noV6uipu7YVLwbb16HbFYWCcrIMhAqoJy5rI8E1cPfNLKxkipA_UKuJXuDcG00AF",
          "干细胞VS免疫细胞": "noV6uipu7YVLwbb16HbFYWG-pc3sXRiIfHbob7hOL_AKHzEy5KTRYWxz1rXK9mT7",
          "糖尿病": "noV6uipu7YVLwbb16HbFYaPeSVbgF2YVuNkmlANgjfNuEtg6kAk-O1Y4Si1UEAMx",
          "癌症": "noV6uipu7YVLwbb16HbFYXKd9m8rHIRJDmZypQ7vtLIAz_hBgVKY8QRLRwHax9Wv",
          "肺病": "noV6uipu7YVLwbb16HbFYcgFYLcRdkb3qhzX3QLVCrYdFNfvXDGFSiFLIbLRWNhH",
          "干细胞": "noV6uipu7YVLwbb16HbFYXtLjpxmeESofvTgSA9NuT223w2qciBmsIV0rnue5XsF",
          "好孕、怀孕、备孕": "noV6uipu7YVLwbb16HbFYelF4g4clOcXSvhh0_DD1Mku_-aDB7g0Ry7zGJtrisa3",
          "医院": "noV6uipu7YVLwbb16HbFYSUwXbLJ_IaR3SMK1SrOEdD4LF5O9TEKV6SyE2U4EKcN"
        },
        "service_ad": "",                        // 服务广告（HTML源码：注意转义防止json识别报错）
        "contact_us": "",                        // 联系我们（HTML源码：注意转义防止json识别报错）
        "bottom_ad_top": "",                     // 底部广告（HTML源码：注意转义防止json识别报错）
        "bottom_ad_bottom": "",                  // 底部广告（HTML源码：注意转义防止json识别报错）
        "followme": ""                           // 公众号关注Banner（HTML源码：从公众号编辑器F12方式获取，注意转义防止json识别报错）
      }
    ]
  }
```

#### 配置字段说明

- **`gpts`**: 包含 GPT 模型的配置，如 API 地址、密钥、模型列表等。
- **`mps`**: 包含微信公众平台的配置，如 AppID、AppSecret、媒体 ID 映射等。
- **`stopk`**: 停用词文件路径，用于分词和关键词提取。

---

## 使用说明

### 运行脚本

1. **确保已安装所有依赖**。

2. **配置 `config.json` 文件**，填写正确的微信公众平台凭据和 GPT 模型信息。

3. 运行脚本：

   ```bash
   python main.py
   ```

### 日志记录

日志信息将输出到控制台，方便调试和监控程序运行状态。

---

## 功能详解

### 1. 获取 Access Token

通过微信公众平台 API 获取 Access Token，用于后续的草稿上传和文章发布操作。

```python
token_status, token = wechat.get_wechatmp_token()
if not token_status:
    logging.error("获取 Access Token 失败")
```

### 2. 上传图片

将本地图片上传至微信公众平台，并返回媒体 ID。

```python
image_status, media_id = upload_wechatmp_image(token, "path/to/image.jpg")
if not image_status:
    logging.error("图片上传失败")
```

### 3. 生成文章内容

通过 AI 模型生成文章标题和内容，并提取 SEO 关键词。

```python
title = Extensions.get_gpt_generation(keyword="人工智能", lang="zh", mode="title")
content = Extensions.get_gpt_generation(keyword=title, lang="zh", mode="body")
tags = Extensions.extract_keywords(content, require_words=["人工智能", "机器学习"])
```

### 4. 保存草稿

将生成的文章保存为微信公众平台的草稿。

```python
draft_status, draft_id = wechat.get_wechatmp_draft(articles)
if not draft_status:
    logging.error("保存草稿失败")
```

### 5. 发布文章

将草稿发布为正式文章。

```python
publish_status, publish_id = wechat.get_wechatmp_publish(draft_id)
if not publish_status:
    logging.error("文章发布失败")
```

---

## 代码结构

```
WeChatMP-AutoPost/
├── main.py                  # 主程序入口
├── config.json              # 配置文件
├── ChineseStopWords.txt     # 中文停用词文件
├── EnglishStopWords.txt     # 英文停用词文件
├── keywords.txt             # 关键词列表
├── required_keywords.txt    # 必要关键词列表
└── article/                 # 生成的文章备份目录
```

---

## 常见问题

### Q: 如何添加新的 GPT 模型？

A: 编辑 `config.json` 文件，在 `gpts` 字段中添加新的模型配置。例如：

```json
{
    "baseurl": "https://new-model-provider.com/api",
    "apikey": ["new-api-key"],
    "models": [],
    "fee_type": "free",
    "from_type": "text",
    "to_type": "text",
    "min_tokens": 4096
}
```

**后四个参数为非必填项，可自行添加。**

### Q: 如何处理图片？

A: 脚本会自动上传图片并返回媒体 ID。确保图片路径正确，或者在 `config.json` 中配置媒体 ID 映射（从微信公众平台通过API获取）。

### Q: 如何调整文章生成逻辑？

A: 修改 `Extensions.get_gpt_generation` 方法中的提示模板（`prompts`），以适应不同的写作风格和需求。

---

## 贡献指南

1. **Fork** 项目到你的 GitHub 账户。

2. **Clone** 你的 Fork 到本地：

   ```bash
   git clone https://github.com/yourusername/WeChatMP-AutoPost.git
   cd WeChatMP-AutoPost
   ```

3. **创建** 一个新的分支：

   ```bash
   git checkout -b feature/your-feature
   ```

4. **提交** 你的更改：

   ```bash
   git add .
   git commit -m "Add your feature"
   ```

5. **Push** 到你的分支：

   ```bash
   git push origin feature/your-feature
   ```
   
6. 打开一个 **Pull Request** 到 `main` 分支。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

希望这个 `README.md` 文档能够帮助用户更好地理解和使用你的项目！如果有任何补充或修改需求，请随时告知。