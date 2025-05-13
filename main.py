#-*- coding: UTF-8 -*-
__author__ = "https://github.com/lopinx"
# 导出包： uv pip freeze | uv pip compile - -o requirements.txt
# =========================================================================================================================
# pip install httpx[http2,http3] keybert scikit-learn jieba nltk rank-bm25 fuzzywuzzy python-Levenshtein markdown pygments pymdown-extensions markdownify openai pandas python-slugify pypinyin tomlkit
# 朱雀大模型检测：https://matrix.tencent.com/ai-detect/
# 朱雀大模型续杯：`localStorage.setItem('fp',Array.from({ length: 32 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join(''))`
#
# ==========================================================================================================================
import json
import logging
import random
import re
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import jieba
import jieba.analyse
import markdown  # markdown转html
import nltk
import numpy as np
import tomlkit
from fuzzywuzzy import process
from keybert import KeyBERT
from markdownify import markdownify  # html转markdown
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.util import everygrams
from openai import OpenAI
from pypinyin import Style, lazy_pinyin
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from slugify import slugify

#  ##########################################################################################################################
# 当前工作目录,配置文件
WorkDIR = Path(__file__).resolve().parent
config = json.load(open(WorkDIR/"config.json", 'r', encoding='utf-8'))
# 下载分词库数据（首次运行需要）
try:
    nltk.corpus.stopwords.words()
except LookupError:
    nltk.download('stopwords')
try:
    nltk.sent_tokenize("Test sentence")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
# markdown库扩展
markdown_extensions = [
    'fenced_code',                                          # 代码块（三个反引号包裹）
    'codehilite',                                           # 代码高亮（需安装 pygments）
    'tables',                                               # 表格支持
    'footnotes',                                            # 脚注
    'sane_lists',                                           # 智能列表格式化
    'nl2br',                                                # 换行转<br>标签
    # 支持缩写（如 [[NASA|National Aeronautics...]],<a title="National Aeronautics and Space Administration" href="http://www.nasa.gov/">NASA</a> ）
    'abbr',                                                 # 支持缩写
    'toc',                                                  # 目录
    # 'strikethrough',                                        # 删除线
    'pymdownx.tasklist'                                     # 任务列表(待办列表（需 pymdown-extensions）)    
]
markdown_extconfigs = {
    'codehilite': {'linenums': True, 'pygments_style': 'monokai'}
}

# 远程 https://res.cdn.issem.cn/ChineseStopWords.txt, 并将内容转换为列表
try:
    cn_stopk, en_stopk = [[*map(str.strip,filter(str.strip,(WorkDIR/config['stopk'][l]).open(encoding='utf-8')))] for l in ('cn','en')]
except:
    cn_stopk, en_stopk = [], []
# 日志记录配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logging.info(f"🚀 启动程序 {__author__}")
# ##########################################################################################################################

class WeChatMP():
    def __init__(self, wechat: Dict, gpts: List[Dict]) -> None:
        self.gpts = gpts
        self.wechat = wechat

    """令牌：返回状态和令牌"""
    def get_wechatmp_token(self) -> Tuple[bool, str]:
        try:
            with (Path(WorkDIR) / f"{self.wechat.get('appid')}.json").open("r", encoding="utf-8") as f:
                data = json.load(f)
            expires_in = data.get("expires_in", 0)
            # (设置提前200秒过期，用来抵扣程序运行中耗时)
            if expires_in > 0 and expires_in > int(time.time()) + 200:        
                return True, data.get("access_token")
        except (FileNotFoundError, json.JSONDecodeError):
            try:
                with httpx.Client() as client:
                    url = f"{self.wechat['baseurl']}/cgi-bin/token"
                    params = {
                        'grant_type': 'client_credential',
                        'appid': self.wechat.get('appid'),
                        'secret': self.wechat.get('secret')
                    } 
                    res = client.get(url, params=params)
                    res.raise_for_status()
                    data = res.json()
                    if "access_token" in data:
                        # 修改过期时间：将返回的过期时长修改为当前时间戳加上返回的过期时长
                        data["expires_in"] = int(time.time()) + data["expires_in"]
                        with (Path(WorkDIR) / f"{self.wechat.get('appid')}.json").open("w", encoding="utf-8") as f:
                            json.dump(data, f)
                        return True, data.get("access_token")
                    else:
                        return False, data.get("errmsg", "")
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                return False, "获取access_token失败"
            

    """草稿:返回状态和文章ID"""
    def get_wechatmp_draft(self, articles: List[Dict]) -> Tuple[bool, str]:
        token = self.get_wechatmp_token()[1]
        if not token[0]:
            return False, "获取access_token失败"
        _drafts = []
        # 上传图片 返回(sucess, media_id)
        @staticmethod
        def upload_wechatmp_image(token: str, file_path: str) -> Tuple[bool, str]:
            try:
                with open(file=file_path, mode='rb') as fp:
                    url = f"{self.wechat['baseurl']}/cgi-bin/material/add_material"
                    params = {
                        'access_token': token,
                        'type': 'image'
                    }         
                    files = {'media': fp}
                    with httpx.Client() as client:
                        res = client.post(url=url, params=params, files=files)
                        res.raise_for_status()
                        try:
                            data = res.json()
                            if "media_id" in data:
                                 return True, data.get("media_id", "")
                            else:
                                 return False, data.get("errmsg", "")
                        except json.JSONDecodeError:
                            return False, "响应内容无法解析为JSON"
            except (httpx.RequestError, httpx.HTTPStatusError, FileNotFoundError, Exception) as e:
                # logging.error()
                return False, f"图片上传失败: {e}"

        for article in articles:
            try:
                # 匹配最相似的键
                media_ids = self.wechat.get("media_ids")
                _titmatch = process.extractOne(article["title"], media_ids.keys())
                image_stauts, cover_media_id = True, (media_ids[_titmatch[0]] if _titmatch else random.choice(media_ids.values()))
            except:
                image_stauts, cover_media_id = upload_wechatmp_image(token, article["cover_path"]) 

            if not image_stauts: return False

            _drafts.append({
                "article_type": "news",
                "title": article["title"],
                "thumb_media_id": cover_media_id,                   # 图文消息的封面图片素材id（必须是永久MediaID）
                "content": article["content"],                      #{company_info}{top_ad}{bottom_ad}
                "content_source_url": self.wechat.get("source_url"),    # 图文消息的原文地址，即点击“阅读原文”后的URL
                "need_open_comment": 1,                             # 是否打开评论，0不打开，1打开
                "only_fans_can_comment": 0,                         # 是否粉丝才可评论，0所有人可评论，1粉丝才可评论
                "author": self.wechat.get("author"),                    # 非必填
                # "digest": "digest",                               # 非必填(仅有单图文消息才有摘要，则默认抓取正文前54个字。)
                # "pic_crop_235_1": "X1_Y1_X2_Y2",                  # 封面图片的裁剪坐标，坐标以图片左上角为原点，x轴向右，y轴向下
                # "pic_crop_1_1": "X1_Y1_X2_Y2"                     # 封面图片的裁剪坐标，坐标以图片左上角为原点，x轴向右，y轴向下
            })

        drafts = {"articles": _drafts}
        # 直接post json.dumps(drafts) 会有乱码    
        post_data = json.dumps(drafts, ensure_ascii=False).encode('utf-8')
        try:
            with httpx.Client() as client:
                url = f"{self.wechat['baseurl']}/cgi-bin/draft/add"
                params = {
                    'access_token': token
                }     
                res = client.post(url=url, params=params, content=post_data)
                res.raise_for_status()
                try:
                    data = res.json()
                    if "media_id" in data:
                        return True, data.get("media_id", "")
                    else:
                        return False, data.get("errmsg", "")
                except json.JSONDecodeError:
                    return False, "响应内容无法解析为JSON"
        except Exception as e:
            return False, f"草稿上传失败: {e}"
        

    """发布:返回状态和文章ID"""
    def get_wechatmp_publish(self, draft_id: int) -> Tuple[bool, str]:
        token = self.get_wechatmp_token()[1]
        if not token[0]:
            return False, "获取access_token失败"
        try:
            with httpx.Client() as client:
                url = f"{self.wechat['baseurl']}/cgi-bin/freepublish/add"
                params = {
                    'access_token': token
                }
                res = client.post(url=url, params=params ,json={"media_id": draft_id})
                res.raise_for_status()
                try:
                    data = res.json()
                    if "media_id" in data and data.get("errcode") == 0:
                        return True, data.get("publish_id", "")
                    else:
                        return False, data.get("errmsg", "")
                except json.JSONDecodeError:
                    return False, "响应内容无法解析为JSON"
        except Exception as e:
            return False, f"文章发布失败：{e}" 


# ========================================================================================================================
# 以下为拓展类，按需使用
# ========================================================================================================================

    """发布到文章到公众号上"""
    def main(self):
        # 1. 获取文章数据
        articles = Extensions.get_article_data(self.wechat)
        if not articles:
            return False
        if self.wechat['draft']:# 2 保存微信草稿
            draft_status, draft_id = self.get_wechatmp_draft(articles)
            if not draft_status: 
                return False
        if self.wechat['publish']:# 3 微信推送发布
            publish_status, publish_id = self.get_wechatmp_publish(draft_id)
            if not publish_status: 
                return False
            elif len(publish_id) > 16:
                return "发布成功", publish_id
            else: 
                return "发布失败", publish_id


class Extensions:
    """获取所有可用GPT配置"""
    @staticmethod
    def get_gpt_config(gpts: List[Dict]) -> List[Dict]:
        new_gpts = []
        for entry in gpts:
            if any('sk-or-' in key for key in entry['apikey']) and len(entry["models"])==0:
                get_models = Extensions.get_openrouter_models(
                    entry["baseurl"],
                    entry["fee_type"], 
                    entry["from_type"], 
                    entry["to_type"], 
                    entry["min_tokens"]
                )
                _entry = {
                    "baseurl": entry["baseurl"],
                    "apikey": entry["apikey"],
                    "models": get_models
                }
                new_gpts.append(_entry)
            else:
                new_gpts.append(entry)
        return new_gpts
    

    """获取所有可用模型列表"""
    @staticmethod
    def get_openrouter_models(
        baseurl: str = "https://openrouter.ai/api/v1", 
        fee_type: str = "free", 
        from_type: str = "text", 
        to_type: str = "text", 
        min_tokens: int = 4096
    ) -> List[Dict]:
        # 获取可用模型列表
        type_models = []
        try:
            with httpx.Client() as client:
                resp = client.get(url=f"{baseurl}/models")
                resp.raise_for_status()
                models_data = resp.json()
        except Exception as e:
            return []

        _free, _paid = [], []
        # 第一步：筛选模型类型
        for _f in models_data.get("data", []):
            pricing = _f.get("pricing")
            if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                _free.append(_f)
            else:
                _paid.append(_f)   
        models_detail = {"paid": _paid, "free": _free}.get(fee_type, _free + _paid)
            
        # 第二步：筛选模型类型
        for model in models_detail:
            modality = model.get("architecture").get("modality")
            max_completion_tokens = model.get("top_provider").get("max_completion_tokens")
            _tokens = max_completion_tokens is None or max_completion_tokens > min_tokens
            _type = from_type + "->" + to_type
            if modality == _type and _tokens:
                # type_models.append(model)
                type_models.append({
                    "baseurl": "https://openrouter.ai/api/v1",
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "pricing": model.get("pricing"),
                    "max_tokens": max_completion_tokens,
                    "context_length": model.get("context_length")
                })
            else:
                continue
        
        # return list(type_models)
        return [models.get("id") for models in type_models]
    
    
    """用分词器提取元关键词"""
    @staticmethod
    def extract_article_keywords(content: str, require_words: List[str] ) -> List[str]:
        if not content: return []
        # 处理成Markdown格式
        try:
            html = markdown.markdown(content)
            if not (('<' in html and '>' in html) and html.strip() != content.strip()):
                content = markdownify(content)
        except Exception:
            content = markdownify(content)
        # 清理非法字符（保留中英文、数字、常见符号）
        # 需要匹配英文单词、中文、专有英文缩写（如 NASA、U.S.A.）以及特定符号
        content = re.sub(
            r'[^a-zA-Z0-9\u4e00-\u9fa5\s\-.\'@#$%&*+/:;=?~(){}$`_、。，《》？！“”‘’（）—…]',
            ' ',                                                # 用空格替代非法字符
            content,                                            # 替换目标字符串
            flags=re.UNICODE
        ).strip()
        # 判断文本语言（中文/英文）
        cn_lang = any(
            (u'\u4e00' <= char <= u'\u9fa5') or
            (u'\u3400' <= char <= u'\u4DBF') or
            (u'\U00020000' <= char <= u'\U0002A6DF')
            for char in content
        )
        # 可选停用词
        if not cn_lang:
            stop_words = set(stopwords.words('english')).union(en_stopk)
        else:
            stop_words = set(stopwords.words('chinese')).union(cn_stopk)
        # ===============================================================================================================
        # Keybert 算法（英文） / TextRank + jieba 算法（中文）
        # ===============================================================================================================
        if not cn_lang:
            # 英文处理：KeyBERT + 语义优先
            kw_model = KeyBERT()
            keywords = kw_model.extract_article_keywords(
                content.lower(),
                keyphrase_ngram_range=(1, 6),
                stop_words=stop_words,
                use_mmr=True,
                diversity=0.5
            )
            vectorizer = TfidfVectorizer(ngram_range=(1, 6))
            tfidf = vectorizer.fit_transform([content])
            vocab = vectorizer.vocabulary_  # 获取词汇表
            # 综合评分：BERT置信度 × TF-IDF
            scores = {
                word: score * tfidf[0, vocab.get(word, -1)]
                for word, score in keywords
                if word in vocab and word.lower() in content.lower()  # 确保关键词在文章中
            }
        else:
            # 中文处理：TextRank + 词密度
            # 将用户词逐个添加到jieba词典（确保短语不被拆分）
            list(map(jieba.add_word, require_words))
            keywords = [word for word in jieba.cut(content) if word not in stop_words or word in require_words]
            vectorizer = TfidfVectorizer(ngram_range=(1, 6), token_pattern=r'[^\s]+')
            tfidf_matrix = vectorizer.fit_transform([' '.join(keywords)])
            vocab = vectorizer.vocabulary_  
            tfidf = tfidf_matrix.toarray()[0]
            # TextRank权重（允许短语）
            text_rank = {k: v for k, v in jieba.analyse.extract_tags(' '.join(keywords), topK=500, withWeight=True, allowPOS=())}
            scores = {
                word: text_rank.get(word, 0) * tfidf[vocab[word]] 
                for word in keywords 
                 if word in vocab and word in content  # 确保关键词在文章中
            }
        # 强制合并用户词典中的词（中英文统一处理）
        for key in require_words:
            word = key.lower()
            if word in content.lower() and word not in scores:  # 确保用户词在文章中
                scores[word] = tfidf[vocab[word]] if cn_lang else 1.0 * tfidf[0, vocab[word]]
        # 强制包含用户提供词（统一逻辑）
        for req_word in (r.lower() for r in require_words):
            if req_word in content.lower():  # 确保用户词在文章中
                current = scores.get(req_word, 0)
                scores[req_word] = current * 2 if current else max(scores.values(), default=0) * 2 + 1

        # 生成排序后的关键词列表
        t_k = sorted(scores.keys(), key=lambda k: (-scores[k], -len(k)))
        # 过滤不符合条件的关键词
        filter_pattern = re.compile(r'^[\W_]+$|^\d+(?:\.\d+)?%?$|^\d+[eE][+-]?\d+$')
        return [k for k in t_k if len(k) >= 2 and not filter_pattern.fullmatch(k)]
        # ===============================================================================================================


    """通过AI生成标题或内容"""
    @staticmethod
    def get_gpt_generation(keyword: str, lang: str = "",mode: str = "body") -> Optional[str]:
        gpt = random.choice(gpts)
        role = f"你是一个具有丰富行业知识{'，深谙中国《广告法》' if lang !='英文' else ''}的资深{lang}文案编辑"
        if mode == "body":
            prompts = f"""请以<{keyword}>为标题，并结合行业特点写一篇{lang}爆款科普性文章

要求：
- 采用文案创作黄金三秒原则。
- 要求站在用户角度思考和解读，直击用户痛点，共情用户情绪，并且引导用户咨询和持续关注。
- 排版需要考虑阅读体验，合理安排重点信息词语或者段落高亮，每行都需要间隔一个空行（代码除外），分割符统一使用20个连续破折号代替。
- 正文引用处用上标标注，文末按APA格式列尾注（作者，年份，标题，期刊，DOI），不得虚构尾注内容。
- 在不破坏阅读体验的情况下在合适位置（段落之外：不得破坏段落结构）安排插入 “[文内插图]” 进行占位，以丰富文章内容。
- 不要出现与内容无关的解释性语句和文章提示：不要解释“我正在生成文章”之类的引导语；不得出现 "标题：" 或类似提示语。

此外，新写的文章需要具备以下特点：
- 文章结构清晰，段落过渡自然，避开冷僻词汇，删减文章末尾总结/结论/展望部分，让内容简洁流畅，便于轻松把握思路与意图。
- 替换或减少结构化连接词（如：首先、其次、然后、再次、最后、总之、总而言之、然而、因此、另外、此外），改用更基础、常用、甚至口语化的表达，以增加自然度和可读性。
- 内容充实，论据充分，支撑观点的同时展现独特视角；辅以案例与数据，提升可信度与说服力（谷歌算法EEAT原则）。
- 风格适配，语言要素（用词/语气/句式/结构）贴合读者需求；融入个人视角与情感，增强文章个性与温度。
- 融合BERT语义建模、TF-IDF向量分析、BM25相关性排序、三元组知识图谱与LDA主题模型，全面提升SEO的语义理解、关键词优化及内容关联性。
- 规避《广告法》禁用词：禁用“最”“唯一”等绝对词；效果描述加“可能”“部分用户反馈”等限定词；医疗/食品类禁用“治疗”“治愈”，改用“辅助”“护理”；数据标注来源（如“内部调研显示…”）。
- 不得引用或者使用虚假网址（如：example.com）和虚假数据（如：品牌、产品、公司、组织、网站、机构、人员、事件、地点、时间、数量、金额、百分比、比例、概率等。
- 写法指南：杜绝标题党（内容需与标题匹配），避免关键词堆砌（防止机械感），拒绝自嗨（测试三人法则：三人超3秒犹豫则重写）。聚焦标题和关键词，切忌不要偏题、跑题。

请严格按照以上要求，只需以Markdown格式输出正文内容，不再需要再输出标题以及其他解释。
"""
        elif mode == "title":
            prompts = f"""请以<{keyword}>为关键词，参考下方10种标题写作方法，并结合行业特点写法拟一个{lang}爆款标题

要求：
- 杜绝使用“：”分割标题，标题末尾不需要句号；
- 使用口语化表达，避免机械感；
- 能激发情绪共鸣（如好奇、焦虑、惊喜、愤怒、惊恐）；
- 制造认知反差或悬念，引导点击；
- 不需要任何解释性语句或副标题；
- 直接返回标题文字，不要加Markdown格式或标签。

以下是10种标题写作方法供参考：
【第一招】数字法进阶版：月薪5千到5万，我用了237天
【第二招】悬念设计法：领导半夜发微信，我整晚没睡着
【第三招】痛点场景化：孩子考不上重点高中？这5个自救方案现在看还来得及
【第四招】情绪过山车：同事据为己有→反击鼓掌；被房东赶出→住月租500豪宅
【第五招】热点降维术：谷爱凌每天睡10小时？打工人学三点就够了
【第六招】认知反差术：越省钱的人越穷，拼命工作的人升不了职
【第七招】灵魂拷问法：为什么你学了100个写作课还是写不好文案？
【第八招】强势指令法：收藏！这20个Excel公式省80%工作量
【第九招】符号节奏术：月入3W+！适合懒人的搞钱副业（附渠道）
【第十招】毛孔级场景：蹲马桶刷手机时顺手就能做的5个搞钱副业

请严格按照要求生成一个标题，只输出标题内容，不需要其他解释、标签或格式。
""" 
        elif mode == "excerpt":
            prompts = f"""请根据我提供的文章内容并结合搜索引擎规则对文章做一个100字左右{lang}简单概述，以纯文本输出概述内容，无需原文和其他，以下是文章内容：\n{keyword}"""
        elif mode == "tags":
            prompts = f"""请根据我提供的文章内容并结合搜索引擎规则提取出5个合适的关键词，用逗号分隔，以纯文本输出关键词内容，无需原文和其他，以下是文章内容：\n{keyword}"""
        else:
            prompts = ''
        client = OpenAI(
            api_key=random.choice(gpt["apikey"]), 
            base_url=gpt["baseurl"],
            http_client=httpx.Client(verify=False)
        )
        try:
            completion = client.chat.completions.create(
                model=random.choice(gpt["models"]),
                messages=[
                    {"role": "system","content": role},
                    {"role": "user","content": prompts}
                ],
                n=1,                    # 输出的条数
                temperature=0.5,        # 输出的随机性，值越低输出越确定
                top_p=0.8,              # 输出的多样性，值越低输出越集中
                max_tokens=4096,        # 控制生成的最大token数量
                frequency_penalty=0.5,  # 减少重复内容的生成
                presence_penalty=0.5    # 鼓励模型引入新内容
            )
            content = completion.choices[0].message.content
            return re.sub(r'<think>.*</think>', '', content.strip(), flags=re.DOTALL)
        except Exception as e:
            logging.error(f"{str(e)}")
            return None
        

    """整合成发布的数据结构"""
    @staticmethod
    def get_article_data(platform: Dict) -> List[Dict]:
        # 文章内容二次精细化处理
        articles = []
        # 获取所有关键词或者标题列表，根据文件名：keywords.txt / titles.txt
        # 当前是关键词模式，还是标题模式
        mode = 'keywords' if 'keywords' in platform.get('keys') else 'titles'
        if '.txt' not in platform.get('keys'):
            keys = list(set(platform.get('keys')))
        elif (key_path := WorkDIR / platform.get('keys')).exists():
            with key_path.open('r', encoding='utf-8') as _key:
                keys = list(set([s for l in _key if (s := re.sub(r'[\n\ufeff]', '', l)) and len(s) > 1]))
        else:
            return []
        # 生成所需文章数据
        for _ in range(platform.get('number', 1)): 
            key = random.choice(keys)                       # 随机选择一个基础关键词 
            # 文章标题生成         
            _title = ''
            if mode == "titles":                            # 纯标题，不需要生成标题
                _title = key
            else:                                           # 纯关键词，需要生成标题
                for _t in range(3):
                    _tit = Extensions.get_gpt_generation(keyword=key, lang=platform.get('lang'), mode="title")
                    if _tit and not (_tit.startswith('<p>') or _tit.startswith('<h3>') or ' error' in _tit):
                        _title = re.sub(r'[\n<>"\'《》{}【】「」——。]|&nbsp;|&lt;|&gt;|boxed', '', _tit).strip(' \n\r"\'')
                        zh_title_len = len(re.findall(r'[\u4e00-\u9fff]', _title))
                        en_title_len = len(re.findall(r"[A-Za-z'-]+", _title))
                        if not ((5 < zh_title_len < 30) or (5 < en_title_len < 30)):
                            logging.error("文章标题 【字数】 不符合要求")
                            continue
                        break
                    else:
                        logging.error(f"文章标题 【内容】 不符合要求")
                        continue
            logging.info(f"{_title}, {len(_title)}")
            
            # 文章内容规则
            for _t in range(3):
                _content = Extensions.get_gpt_generation(keyword=_title, lang=platform.get('lang'), mode="body")
                if _content and len(_content) > 100:
                    zh_content_len = len(re.findall(r'[\u4e00-\u9fff]', _content))
                    en_content_len = len(re.findall(r"[A-Za-z'-]+", _content))
                    if not (zh_content_len > 300 or en_content_len > 300):
                        logging.error(f"文章详情 【字数】 不符合要求")
                        continue
                    # 替换分隔符：统一合并为20个符号
                    _content = re.sub(
                        r'^([=-]{3,})[\r\n]+', 
                        lambda m: '='*20 + '\n' if m.group(1).startswith('=') else '-'*20 + '\n',
                        _content,
                        flags=re.MULTILINE
                    )
                    # 尝试多种方式匹配并移除标题
                    # 情况1：第一行为标题(或者带#号) -> 移除第一行
                    # 情况2：标题被拆分为两行，如 “第一部分\n第二部分” 或者 “第一部分\n——第二部分” -> 移除前两行
                    # 最后处理剩下的内容：删除以 '——' 开头的首行（如有）
                    lines = _content.split('\n')
                    if lines and (lines[0] == _title or re.fullmatch(rf'#\s+{re.escape(_title)}', lines[0])):
                        lines = '\n'.join(lines[1:]) 
                    elif len(lines) >= 2 and ((lines[0] + lines[1]).strip() == _title or re.match(rf'^—+\s+{re.escape(_title.split()[-1])}$', lines[1])):
                        lines = '\n'.join(lines[2:]) 
                    else:
                        pass
                    if lines and lines[0].startswith('——'): _content = '\n'.join(lines[1:])
                    break
                else:
                    logging.error(f"文章详情 【内容】 不符合要求")
                    continue
            
            # 提取文章标签
            if not platform.get('aitags'):
                if '.txt' not in platform.get('reqkeys'):
                    reqkeys = list(set(platform.get('reqkeys')))
                elif (reqkeys_path := WorkDIR / platform.get('reqkeys')).exists():
                    with reqkeys_path.open('r', encoding='utf-8') as req_key:
                        reqkeys = list(set([s for l in req_key if (s := re.sub(r'[\n\ufeff]', '', l)) and len(s) > 1]))
                else:
                    reqkeys = keys if mode == 'keywords' else []
                _tags = Extensions.extract_article_keywords(_content, reqkeys)[:5]
            else:
                _ts = Extensions.get_gpt_generation(keyword=_content, lang=platform.get('lang'), mode="tags")
                _tags = [x.strip() for x in re.split(r'[,\u3001]+', _ts) if x.strip()]
            
            # 提取文章摘要
            _excerpt = Extensions.get_gpt_generation(keyword=_content, lang=platform.get('lang'), mode="excerpt")
            
            # 提取文章图片
            _img = f"""![{key}](https://image.pollinations.ai/prompt/{key}?width=1200&height=900&enhance=true&private=true&nologo=true&safe=true&model=flux "{key}")"""
            _content = _content.replace('[文内插图]', _img)
            _pictures = list(re.compile(r'!\[.*?\]\(\s*([^\s)]+)(?:\s+|\))').findall(_content))
           
            # 处理文章内容
            _content = Extensions.handle_article_content(_title, _content, platform)

            # 构建文章数据字典
            article = {
                "title": _title,
                "keyword": key,
                "content": _content,
                "tags": _tags if _tags else [key],
                "excerpt": _excerpt,
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                "pictures": _pictures,
                "cover": _pictures[:1] if _pictures else '',
            }
            # 判断文章是否为空(所有字段)
            # if not any(v == "" or v is None or (hasattr(v, '__len__') and len(v) == 0) for v in article.values()):
            # 判断文章是否为空(排除列表)
            if not any(v == "" or v is None or (not isinstance(v, list) and hasattr(v, "__len__") and len(v) == 0) for v in article.values()):
                # 添加到articles列表
                articles.append(article)
                # 导出到Markdown文件
                Extensions.export_to_markdown(platform, article)
        return articles

    
    """整合其他信息到文章中"""
    @staticmethod
    def handle_article_content(title: str, content: str, platform: Dict) -> str:
        # 1. 定义 Markdown 特征的正则表达式
        markdown_patterns = [
            r'^#{1,6}\s+',                    # 标题（如 #, ##）
            r'^[-*]\s+',                       # 无序列表（- 或 * 开头）
            r'^\d+\.\s+',                      # 有序列表（数字. 开头）
            r'\*\*.*?\*\*|\*.*?\*',            # 粗体 (**text**) 或斜体 (*text*)
            r'!\[.*?\]\(.*?\)|\[.*?\]\(.*?\)', # 图片或链接
            r'^```.*?^```',                    # 代码块（多行）
            r'`.*?`',                          # 行内代码（`text`）
            r'^>\s+',                          # 引用（> 开头）
            r'^\|.*?\|',                       # 表格（|...|）
            r'^[=-]{3,}\s*$',                  # 分割线（--- 或 ===）
            r'^\s*[*-]{3,}\s*$',               # 单独的分割线（---, ***）
        ]
        # 2. 统计有效特征数量
        count = 0
        # 3. 预处理：移除代码块和行内代码
        _text = re.sub(r'(?s)`.*?`', '', re.sub(r'(?s)```.*?```|~~~.*?~~~', '', content))
        lines = list(map(str.strip, _text.split('\n')))
        for line in lines:
            if not line: continue
            for pattern in markdown_patterns:
                if re.search(pattern, line, flags=re.MULTILINE | re.DOTALL):
                    count += 1
                    break
        # 4. 判定是否是Markdown内容
        if count >= 2:
            # 使用分隔符分割文本，去除标题，只取后半部分纯文章内容
            # content = re.sub(r'^(={3,}[\r\n]+)', '====================\n', content, flags=re.MULTILINE)
            # content = re.sub(r'^(-{3,}[\r\n]+)', '--------------------\n', content, flags=re.MULTILINE)
            # 替换分隔符：统一合并为20个符号
            content = re.sub(
                r'^([=-]{3,})[\r\n]+', 
                lambda m: '='*20 + '\n' if m.group(1).startswith('=') else '-'*20 + '\n',
                content,
                flags=re.MULTILINE
            )
            # 分隔标题与内容
            parts1 = content.split(f"====================\n", 1)  # 最多分割一次
            parts2 = content.split(f"--------------------\n", 1)  # 最多分割一次
            if len(parts1) > 1 and parts1[0].strip(' \n\r"\'').replace('# ', '', 1) == title:
                content = parts1[1].strip()
            elif len(parts2) > 1 and parts2[0].strip(' \n\r"\'').replace('# ', '', 1) == title:
                content = parts2[1].strip()
            else:
                lines = [line for line in content.strip(' \n\r"\'').splitlines() if line.strip()]
                if lines[0].strip(' \n\r"\'') == title:
                    content = content.replace(lines[0], '', 1).strip()

        # 移除所有级别的总结类标题（如：结语、总结、小结、结论）
        zjbt = r'^(?:#{2,6})\s*(?:结语|总结|小结|结论|总而言之|综上所述|未来展望|展望未来|写在最后)\s*$\n?'
        wmjz = r'(?:[⁰¹²³⁴⁵⁶⁷⁸⁹]+\.)?\s*(?:标题|题目|文献|参考)?(脚注|注脚)：["“]([^，]+)，(\d{4})，([^，]+)，([^，]+)，(10\.\d{4}\/[^"”]+)["”]' 
        content = re.sub(fr'({zjbt}|{wmjz})', '', content, flags=re.MULTILINE | re.IGNORECASE)
        desc = r'^.*?【?(?:本文|文章|本篇|全文|前言)\s*[，,]?\s*(?:简介|摘要|概述|导读|描述|引言)】?[：:]?'
        content = markdown.markdown(
            re.sub(desc,'', content, flags=re.DOTALL | re.MULTILINE), # 去掉机械式开头
            output_format='html',
            extensions=markdown_extensions,
            extension_configs=markdown_extconfigs,
            safe_mode='escape',
            enable_attributes=True,
            linkify=True,
            tab_length=4,
            lazy_ol=False,
            toc=True,
            toc_depth=4,
            toc_title=u'文章目录',
            toc_nested=True,
            toc_list_type='ul',
            toc_list_item_class='toc-list-item',
            toc_list_class='toc-list',
            toc_header_id='toc',
            toc_anchor_title=u'跳转至文章目录',
            toc_anchor_title_class='toc-anchor-title'
        )
        # 匹配所有<h3>、<p>标签，设置样式
        content = re.sub(r'<p>', platform.get('phtag'), content)
        content = re.sub(r'<h3>', platform.get('hrtag'), content)
        # 文章再处理（符合平台需要）有BUG
        # content  = f"""
        #     {platform.get('service_ad')}
        #     {content}
        #     {platform.get('bottom_ad_top')}
        #     {platform.get('contact')}
        #     {platform.get('bottom_ad_bottom')}
        #     {platform.get("followme")}
        # """
        return content


    # 生成CMS文章Dict
    def export_to_markdown(platform: Dict, data: Dict) -> Optional[bool]:
        # 生成Front Matter
        # 将中文部分转换为拼音（保留英文和数字）
        cn_lang = any(
            (u'\u4e00' <= char <= u'\u9fa5') or
            (u'\u3400' <= char <= u'\u4DBF') or
            (u'\U00020000' <= char <= u'\U0002A6DF')
            for char in data.get('title', '')
        )
        if cn_lang:
            _title = '-'.join(lazy_pinyin(data.get('title', ''), style=Style.NORMAL, strict=False))
        else:
            _title = data.get('title', '')
        urlname = slugify(
            _title,
            separator='-',
            lowercase=True,
            regex_pattern=None,
            word_boundary=True,
            stopwords=[],
            replacements=[]
        )
        # 创建 TOML 文档对象
        doc = tomlkit.document()
        # 基础字段
        doc["title"] = data.get('title')
        doc["date"] = data.get('date')
        doc["tags"] = data['tags'][:5]
        doc["keywords"] = data['tags'][:5]
        doc["description"] = data.get('excerpt') or ""
        doc["categories"] = platform.get('categories')  or []
        doc["author"] = platform.get('author') or "lopins"
        doc["cover"] = data.get('cover') or ""
        doc["pictures"] = data.get('pictures') or []
        doc["hiddenFromHomePage"] = False
        doc["readingTime"] = True
        doc["hideComments"] = True
        doc["isCJKLanguage"] = True
        doc["slug"] = urlname
        # 处理扩展字段添加到文档
        _extras = {}
        for k, v in data.get('extras', {}).items():
            if isinstance(v, str):
                v = v.strip().strip('(（）)')
                p = int(v) if v.isdigit() else v
            else:
                p = v
            _extras[k] = p if isinstance(p, int) else f'{json.dumps(p,ensure_ascii=False)[1:-1]}'
        for key, value in _extras.items():
            doc[key] = value
        # 文章状态
        doc["draft"] = False
        # 序列化为 TOML 字符串（保留格式）
        front_matter_block = f"+++\n{tomlkit.dumps(doc).strip()}\n+++"

        if platform.get('cms') == 'hexo':
            content = front_matter_block.strip()[3:-3].strip()  # 直接去除+++分隔符
            lines = content.split('\n')
            _yaml = []
            for line in lines:
                line = line.strip()
                if not line: continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    # 处理列表项（移除方括号并分割）
                    items = [i.strip().strip("'\"") for i in val[1:-1].split(',')]
                    _yaml.append(f"{key}:")
                    _yaml.extend(f"  - {item}" for item in items if item)
                elif val in ('true', 'false'):
                    _yaml.append(f"{key}: {val}")
                else:
                    _yaml.append(f"{key}: {val.strip('\"')}")
            front_matter_block = f"---\n{'\n'.join(_yaml)}\n---"
        try:
            doc_name = f"{re.sub(r'\D', '', data.get('date'))}-{uuid.uuid4()}.md"
            file_path = Path(WorkDIR) / "articles" / doc_name
            file_path.parent.mkdir(parents=True, exist_ok=True) 
            with file_path.open("w", encoding="utf-8") as f:
                f.write(f"{front_matter_block}\n\n{markdownify(data['content'])}")
            return True if file_path.exists() and file_path.stat().st_size > 0 else False
        except Exception as e:
            logging.error(f"保存文件 {file_path} 失败: {str(e)}")
            return False    


if __name__ == "__main__":
    # 获取所有大模型配置
    gpts = Extensions.get_gpt_config(gpts=config.get('gpts'))
    # logging.info(f"获取所有可用GPT配置：{gpts[0]["models"]}")
    if gpts: 
        for wechat in config.get('mps'):
            WeChatMP(wechat, gpts).main()
