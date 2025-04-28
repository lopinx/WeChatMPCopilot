#-*- coding: UTF-8 -*-
__author__ = "https://github.com/lopinx"
# 导出包： uv pip freeze | uv pip compile - -o requirements.txt
# =========================================================================================================================
# pip install httpx[http2,http3] keybert scikit-learn jieba nltk rank-bm25 fuzzywuzzy python-Levenshtein markdown pygments pymdown-extensions markdownify openai pandas
# ==========================================================================================================================
import json
import logging
import random
import re
import time
import uuid
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import jieba
import jieba.analyse
import markdown  # markdown转html
import nltk
import numpy as np
from fuzzywuzzy import process
from keybert import KeyBERT
from markdownify import markdownify  # html转markdown
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.util import everygrams
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

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
cn_stopk, en_stopk = [[*map(str.strip,filter(str.strip,(WorkDIR/config['stopk'][l]).open(encoding='utf-8')))] for l in ('cn','en')]
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
                return "发布失败", None
        return "发布成功", publish_id


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
    def extract_keywords(content: str, require_words: List[str] ) -> List[str]:
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
            keywords = kw_model.extract_keywords(
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


    @staticmethod
    def extract_excerpt(content: str, length: int = 3) -> str:
        # 先将markdown格式转换为纯文本格式
        sentences = sent_tokenize(markdownify(content))
        # 判断文本语言（中文/英文）
        cn_lang = any(
            (u'\u4e00' <= char <= u'\u9fa5') or
            (u'\u3400' <= char <= u'\u4DBF') or
            (u'\U00020000' <= char <= u'\U0002A6DF')
            for char in content
        )
        # 分句处理
        if not cn_lang:
            sentences = sent_tokenize(content)
            stop_words = set(stopwords.words('english')).union(en_stopk)
        else:
            sentences = [s.strip() for s in re.split(r'[。！？\.\!\?]\s*', content) if s.strip()]
            stop_words = set(stopwords.words('chinese')).union(cn_stopk)
        # 分词并去除停用词
        _sents = []
        for sent in sentences:
            if not cn_lang:
                tokens = [word for word in word_tokenize(sent.lower()) if word not in stop_words]
            else:
                tokens = [word for word in jieba.cut(sent) if word not in stop_words]
            _sents.append(tokens)
        # 计算 BM25
        bm25 = BM25Okapi(_sents)
        # 计算每个句子的得分
        scores = []
        for query in _sents:
            scores.append(bm25.get_scores(query).mean())  # 使用平均得分
        # 获取得分最高的句子索引并返回摘要
        excerpt = ' '.join([sentences[i] for i in sorted(np.argsort(scores)[::-1][:length])])
        return excerpt


    """通过AI生成标题或内容"""
    @staticmethod
    def get_gpt_generation(keyword: str, lang: str = "",mode: str = "body") -> Optional[str]:
        gpt = random.choice(gpts)
        role = f"你是一个精通相关领域内知识和技能的{lang}资深文案编辑。"
        if mode == "body":
            prompts = f"""以<{keyword}>为标题，写一篇{lang}爆款科普性文章，以Markdown格式源码返回,只要输出文章内容，不要输出标题。
采用文案创作黄金三秒原则。
要求站在用户角度思考和解读，直击用户痛点，共情用户情绪，并且引导用户关注和咨询。
文章中需要安排放置插图的地方请以 “[文章插图]” 提示（单独一行），我会在后续处理。
文内二维码图片我会使用其他生成，请不要再在文章中提示放置。
排版需要考虑阅读体验，合理安排重点信息词语或者段落高亮，每行都需要间隔一个空行（代码除外）。
不要出现与内容无关的解释性语句和文章提示。
此外，新写的文章需要具备以下特点：
1、结构清晰明了，内容之间过渡自然，读者可以轻松理解文章的思路。
2、包含充足的信息和证据，能够支撑文内观点和论据，同时具有独特的见解和观点。
3、使用简练、准确、明确的语言，语法正确，拼写无误，让读者可以轻松理解文章意图。
4、风格适当，包括用词、语气、句式和结构，适应读者的背景和阅读目的。
5、避免过于学术化的表达，避免机械化的表达，避免使用过时的技术、方法、术语等。
6、引入BERT语义拓扑建模、TF-IDF向量分析、三元组知识图谱和LDA主题模型等高级技术进一步提升SEO效果。
7、文章写法指南：杜绝标题党（内容需与标题匹配），避免关键词堆砌（防止机械感），拒绝自嗨（测试三人法则：三人超3秒犹豫则重写）。
8、请紧扣标题和关键词，切忌不要偏题、跑题。
"""
        elif mode == "title":
            prompts = f"""以<{keyword}>为关键词，请参考以下写法并结合行业特点写法拟一个{lang}爆款标题，请不要使用‘：’分隔标题，要尽量人性化和能勾起读者兴趣，你只需要把标题输出来，无需其他信息。
以下是一些标题写法参考(按照这个句式，“：”前不是标题内容，而是要突出的核心思想)：
- 第一招：数字法进阶版
用具体天数制造真实感：月薪5千到5万，我用了237天
数字组合构建故事感：面试被拒8次后，我悟出了这3个潜规则
要点：奇数字数 > 偶数字数；具体数值 > 笼统整数
- 第二招：悬念设计法
冲突前置+情绪后置：领导半夜给我发了条微信，我整晚没睡着
连续剧式悬念：出差回来发现梳妆台上多了支口红，查完监控我哭了
要点：前半句埋冲突，后半句给情绪出口
- 第三招：痛点场景化
升学焦虑：孩子考不上重点高中？这5个自救方案现在看还来得及
中年危机：35岁被裁员，靠这三个野路子我反而多赚了20万
要点：用「凌晨三点改PPT」替代「职场压力」，用「辅导作业心梗」替代「教育焦虑」
- 第四招：情绪过山车
憋屈→解气：同事把我做的方案据为己有，我的反击让全公司鼓掌
绝望→惊喜：被房东赶出来的第7天，我住进了月租500的豪宅
要点：单次触发一个情绪按钮（愤怒/恐惧/好奇/优越感）
- 第五招：热点降维术
冬奥案例：谷爱凌每天睡10小时？打工人学这三点就够了
热播剧嫁接：《狂飙》高启强混社会的三个底层逻辑，用在职场真香
要点：提取热点核心情绪而非直接评论
- 第六招：认知反差术
薪资对比：月薪3000和月薪3万的人，差的不只是钱
反常识暴击：天天加班的人反而最先被裁？老板不会说的潜规则
制造反逻辑冲突：越省钱的人越穷、拼命工作的人升不了职
- 第七招：灵魂拷问法
扎心提问：为什么你学了100个写作课还是写不好文案？
窥探式设问：你知道老板最怕员工问哪三个问题吗？
预设读者已遇到的问题：为什么越努力越焦虑？
- 第八招：强势指令法
明确行动：收藏！这20个Excel公式能省你80%工作量
危机警告：立刻停止！这五种早餐越吃越胖
要点：搭配必须/千万/立刻/马上+具体利益点
- 第九招：符号节奏术
加号制造想象：月入3W+！适合懒人的搞钱副业（附渠道）
问号引发好奇：95后女生独居vlog爆火，原来拍视频这么简单？
要点：括号补充信息量，感叹号慎用
- 第十招：毛孔级场景
具体场景：蹲马桶刷手机时，顺手就能做的5个搞钱副业
对话还原：领导说'辛苦了'千万别回'应该的'
锁定周五下班前、改第五版方案、微信第3条消息等细节
""" 
        elif mode == "excerpt":
            prompts = f"""请根据我提供的文章内容并结合搜索引擎规则对文章做一个100字左右{lang}简单概述，并以纯文本返回给我，只要输出概述内容，无需原文和其他，以下是文章内容：\n{keyword}"""
        elif mode == "tags":
            prompts = f"""请根据我提供的文章内容并结合搜索引擎规则提取出5个合适的关键词，以逗号分隔，并以纯文本返回给我，只要输出关键词内容，无需原文和其他，以下是文章内容：\n{keyword}"""
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
            # try:
            #     reasoning_content = completion.choices[0].message.reasoning_content
            # except:
            #     reasoning_content = ''
            # 去掉思维链
            return re.sub(r'<think>.*</think>', '', content, flags=re.DOTALL)
        except Exception as e:
            logging.error(f"{str(e)}")
            return None
        

    """整合成发布的数据结构"""
    @staticmethod
    def get_article_data(platform) -> List[Dict]:
        # 文章内容二次精细化处理
        @staticmethod
        def handle_article_content(content: str) -> str:
            # 匹配所有<h3>、<p>标签，设置样式
            content = re.sub(r'<p>', '<p style="box-sizing: border-box; border-width: 0px; border-style: solid; border-color: hsl(var(--border)); margin: 1.5em 8px; text-align: left; line-height: 1.75; font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif; font-size: 14px; letter-spacing: 0.1em; color: rgb(63, 63, 63); visibility: visible;">', content)
            content = re.sub(r'<h3>', '<hr style="box-sizing: border-box; border-width: 2px 0px 0px; border-style: solid; border-color: rgba(0, 0, 0, 0.1); height: 0.4em; color: inherit; margin: 1.5em 0px; text-align: left; line-height: 1.75; font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif; font-size: 14px; transform-origin: 0px 0px; transform: scale(1, 0.5); visibility: visible;">\n<h3>', content)
            # # 找到所有<p>标签的位置，并在中间的</p>后插入内容
            # p_tags = [m.end() for m in re.finditer(r'</p>', content)]
            # if p_tags:
            #     mid_insert_pos = p_tags[len(p_tags) // 2]
            #     content = content[:mid_insert_pos] + bottom_ad + content[mid_insert_pos:]
            # 对<h3>标签进行处理：加粗、下划线和居中
            content = re.compile(r'(</p>\s*)(<p>)', re.DOTALL).sub(
                lambda m: f'{m.group(1)}<p><br /></p>{m.group(2)}',
                content
            )
            content = re.sub(r'<h3>(.*?)</h3>', r'<h3 style="text-align: center;font-size: 1.17em; font-weight: bold; margin-block: 1em;"><strong><span style="text-align:center;display:inline-blocktext-decoration:underline;">\1</span></strong></h3>', content)

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
        # while True:
        for _ in range(platform.get('number', 1)): 
            key = random.choice(keys)                       # 随机选择一个基础关键词          
            _title = ''
            if mode == "titles":                            # 纯标题，不需要生成标题
                _title = key
            else:                                           # 纯关键词，需要生成标题
                for _t in range(3):
                    _tit = Extensions.get_gpt_generation(keyword=key, lang=platform.get('lang'), mode="title")
                    if _tit and not (_tit.startswith('<p>') or _tit.startswith('<h3>') or ' error' in _tit):
                        _title = re.sub(r'[\n<>"\'《》{}【】「」——。]|&nbsp;|&lt;|&gt;|boxed', '', _tit)
                        if not (5<len(re.findall(r'[\u4e00-\u9fff]', _title))<30 or 5<len(re.findall(r"[A-Za-z'-]+", _title))<30):
                            logging.error(f"文章标题 【字数】 不符合要求")
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
                    if not (len(re.findall(r'[\u4e00-\u9fff]', _content))>300 or len(re.findall(r"[A-Za-z'-]+", _content))>300):
                        logging.error(f"文章内容 【字数】 不符合要求")
                        continue
                    break
                else:
                    logging.error(f"文章详情 【内容】 不符合要求")
                    continue
            
            # 提取文章SEO关键词
            # 分词库：必要词
            if not platform.get('aitags'):
                if '.txt' not in platform.get('reqkeys'):
                    reqkeys = list(set(platform.get('reqkeys')))
                elif (reqkeys_path := WorkDIR / platform.get('reqkeys')).exists():
                    with reqkeys_path.open('r', encoding='utf-8') as req_key:
                        reqkeys = list(set([s for l in req_key if (s := re.sub(r'[\n\ufeff]', '', l)) and len(s) > 1]))
                else:
                    reqkeys = keys if mode == 'keywords' else []
                _tags = Extensions.extract_keywords(_content, reqkeys)[:5]
            else:
                _ts = Extensions.get_gpt_generation(keyword=_content, lang=platform.get('lang'), mode="tags")
                _tags = [x.strip() for x in re.split(r'[,\u3001]+', _ts) if x.strip()]
            if not platform.get('aiexcerpt'):
                _excerpt = Extensions.extract_excerpt(content=_content, length=5)
            else:
                _excerpt = Extensions.get_gpt_generation(keyword=_content, lang=platform.get('lang'), mode="excerpt")

            # 去掉机械式开头
            desc_preg = r'^.*?【?(?:本文|文章|本篇|全文|前言)\s*[，,]?\s*(?:简介|摘要|概述|导读|描述)】?[：:]?'
            # 检测Markdown的常见语法，如果不是则进行转换
            if re.search(r'#{1,6} |^-.*$|^```|^\|', _content, re.MULTILINE):
                _html = markdown.markdown(_content)
                if '<' in _html and '>' in _html and _html != _content:
                    _content = markdown.markdown(
                        re.sub(desc_preg,'', _content, flags=re.DOTALL | re.MULTILINE), 
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

            # 文章再处理（符合平台需要）有BUG
            # _content  = f"""
            #     {platform.get('service_ad')}
            #     {handle_article_content(_content)}
            #     {platform.get('bottom_ad_top')}
            #     {platform.get('contact')}
            #     {platform.get('bottom_ad_bottom')}
            #     {platform.get("followme")}
            # """

            article = {
                "title": _title,                        # 文章标题
                "keyword": key,                         # 原始关键词
                "content": _content,
                "tags": _tags if _tags else [key],
                "excerpt": _excerpt
            }
            # 判断文章是否为空
            if not any(v == "" or v is None or (hasattr(v, '__len__') and len(v) == 0) for v in article.values()):
                articles.append(article)
                # 备份文章到本地
                file_path = Path(WorkDIR) / "article" / f"{uuid.uuid4()}.md"  
                file_path.parent.mkdir(parents=True, exist_ok=True) 
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(article.get('content'))
                # else:
                #     logging.error(f"字典中存在空值，跳过该文章")
                #     continue
        return articles
    

if __name__ == "__main__":
    # 获取所有大模型配置
    gpts = Extensions.get_gpt_config(gpts=config.get('gpts'))
    # logging.info(f"获取所有可用GPT配置：{gpts[0]["models"]}")
    if gpts: 
        for wechat in config.get('mps'):
            WeChatMP(wechat, gpts).main()
