# MedicalGPT 工具源码精读 06：merge_tokenizers.py 与 build_domain_tokenizer.py

## 整体作用

这两个脚本和 tokenizer 有关：

- `merge_tokenizers.py`：把中文词表合并进已有 tokenizer。
- `build_domain_tokenizer.py`：训练或构建领域 tokenizer。

对当前 Qwen3 医疗 SFT 主线来说，**不建议先改 tokenizer**。原因是 Qwen3 的 tokenizer 已经很成熟，改词表会牵涉：

- embedding resize
- LoRA 合并
- 推理兼容
- 评估一致性

所以这部分先作为源码学习，不作为当前训练主线。

## merge_tokenizers.py 源码精读

### 判断中文字符

```python
def is_chinese(uchar):
    return '\u4e00' <= uchar <= '\u9fa5'
```

解释：

这通过 Unicode 范围判断单个字符是否是常见汉字。

### 判断中文字符串

```python
def is_chinese_string(string):
    return all(is_chinese(char) for char in string)
```

解释：

如果字符串里每个字符都是中文，就返回 True。

### 加载 Baichuan 词表

```python
def load_baichuan_vocab(vocab_file):
    vocab = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip().split('\t')[0]
            if is_chinese_string(token):
                vocab.append(token)
    return vocab
```

解释：

- 逐行读取词表。
- 取每行 tab 前面的 token。
- 只保留纯中文 token。

### 加载 jieba 词表

```python
def load_jieba_vocab(jieba_vocab_file):
    vocab = []
    with open(jieba_vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip().split(' ')[0]
            if is_chinese_string(token):
                vocab.append(token)
    return vocab
```

解释：

jieba 词典每行通常是：

```text
词 频率 词性
```

所以取空格前的词。

## build_domain_tokenizer.py 学习重点

这个脚本通常用于基于领域语料构建 tokenizer。学习重点不是马上使用，而是理解：

- tokenizer 不是模型本身，但会影响模型输入切分。
- 新增 tokenizer token 后，模型 embedding 也要 resize。
- 如果训练时改 tokenizer，推理和合并时必须使用同一个 tokenizer。

## 和当前项目的关系

当前项目建议：

```text
Qwen3 tokenizer 原样使用
```

不要为了医疗领域词立即扩词。原因是你的目标是复现 SFT / QLoRA / C-Eval 提升，而不是研究 tokenizer 扩词。

后续可以做一个单独实验：

```text
Qwen3 原 tokenizer
vs
医学词表扩展 tokenizer
```

但那是后续研究点。

## 常见坑

- 改 tokenizer 后忘记 `resize_token_embeddings`。
- 训练和推理 tokenizer 不一致。
- 合并 LoRA 时 tokenizer 路径不一致。
- 评估时 tokenizer 变了，导致和 baseline 不可比。

