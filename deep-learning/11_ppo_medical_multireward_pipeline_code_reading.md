# MedicalGPT 训练源码精读 11：PPO 多维奖励函数完整链路

## 整体目标

这一阶段要复现简历里的“PPO 强化学习 + 多维奖励函数”：

```text
SFT 后 Qwen3 医疗模型
  -> 从 SFT-100k 派生 5K 复杂病例 PPO 数据
  -> 模型在线生成回答
  -> 手写格式分、准确率分、安全分
  -> PPO 根据 reward 更新 policy
  -> 提升复杂病例格式合规率和安全回答稳定性
```

这一步和 SFT 不一样。SFT 是给模型标准答案，让它学习“应该怎么回答”；PPO 是让模型先生成回答，再根据 reward 判断这次回答好不好。这里没有额外训练神经网络 reward model，而是用 rule-based reward function 打分。

但是它仍然可以叫 PPO，因为 PPO 需要的是 reward signal，不一定要求 reward 一定来自神经网络模型。严格 PPO 训练里通常有四类组件：

```text
policy model：当前被训练的模型
reference model：冻结参考模型，用于 KL 约束
value model：估计价值，用来算 advantage
reward function / reward model：给回答打分
```

当前实现中：

- `policy model`：SFT 后合并模型，例如 `qwen3_4b_medical_qlora_top100k_ckpt1000_merged`
- `reference model`：同一个 SFT 后模型，冻结使用
- `value model`：Qwen3 backbone + 手写 `ValueScoreHead`
- `reward`：手写多维规则奖励函数

## 整体数据流

```text
data/sft/shibing624_medical_top100k.jsonl
  -> scripts/build_medical_ppo_dataset.py
  -> data/rl/medical_complex_cases_5k.jsonl
  -> MedicalGPT/training/ppo_medical_multireward.py
  -> outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000
```

PPO 数据不是普通 SFT 数据。普通 SFT 数据只有：

```json
{"instruction": "问题", "input": "", "output": "答案"}
```

PPO 多维奖励数据需要保留更多 reward 计算所需信息：

```json
{
  "prompt": "复杂病例问题",
  "reference_answer": "参考答案",
  "answer_keywords": ["关键词1", "关键词2"],
  "risk_level": "high",
  "required_sections": ["病情分析", "处理建议", "风险提示", "就医建议"]
}
```

字段含义：

- `prompt`：PPO 时喂给 policy model 的用户问题。
- `reference_answer`：从原 SFT `output` 转来，用来辅助准确率奖励。
- `answer_keywords`：从参考答案里抽取的医学关键词，用来做关键词覆盖率。
- `risk_level`：是否高风险病例，高风险回答必须更强调就医或医生评估。
- `required_sections`：格式分要求模型回答包含的固定段落。

## 数据集构建脚本

数据构建脚本是：

```text
scripts/build_medical_ppo_dataset.py
```

它的输入是：

```text
data/sft/shibing624_medical_top100k.jsonl
```

输出是：

```text
data/rl/medical_complex_cases_5k.jsonl
data/rl/medical_complex_cases_5k_report.json
```

### 关键源码：`candidate_terms()`

#### 源码

```python
def candidate_terms(text: str) -> list[str]:
    """Extract simple medical keyword candidates without external NLP packages."""
    chunks = re.split(r"[，。；;、,.：:\s（）()【】\[\]<>《》/\\]+", text)
    terms: list[str] = []
    for chunk in chunks:
        chunk = clean_text(chunk)
        if not chunk:
            continue
        if chunk in STOPWORDS:
            continue
        if len(chunk) < 2 or len(chunk) > 16:
            continue
        if re.fullmatch(r"\d+", chunk):
            continue
        terms.append(chunk)

    for hint in MEDICAL_HINT_TERMS + HIGH_RISK_TERMS:
        if hint in text and hint not in terms:
            terms.append(hint)

    counts = Counter(terms)
    ranked = sorted(counts, key=lambda item: (-counts[item], len(item)))
    return ranked[:8]
```

#### 这段代码在做什么

这个函数从参考答案里抽取医学关键词，作为 PPO 准确率奖励的依据。它没有用 jieba 或大模型，而是用规则切分和医学提示词补充，保证脚本在服务器上不依赖额外 NLP 包。

#### 逐段解释

第一段：

```python
chunks = re.split(r"[，。；;、,.：:\s（）()【】\[\]<>《》/\\]+", text)
```

用中文逗号、句号、顿号、空格、括号等符号切分文本。比如：

```text
根据临床表现、心电图、肌钙蛋白检查诊断
```

会拆成：

```text
根据临床表现
心电图
肌钙蛋白检查诊断
```

第二段过滤无效词：

```python
if chunk in STOPWORDS:
    continue
if len(chunk) < 2 or len(chunk) > 16:
    continue
if re.fullmatch(r"\d+", chunk):
    continue
```

它会去掉太短、太长、纯数字和常见泛化词，避免关键词变成“什么”“患者”“疾病”这种没有奖励意义的词。

第三段补充医学提示词：

```python
for hint in MEDICAL_HINT_TERMS + HIGH_RISK_TERMS:
    if hint in text and hint not in terms:
        terms.append(hint)
```

如果答案里含有“心电图”“肌钙蛋白”“胸痛”“卒中”等医学词，即使前面切分没切出来，也强行加入关键词。

最后排序：

```python
counts = Counter(terms)
ranked = sorted(counts, key=lambda item: (-counts[item], len(item)))
return ranked[:8]
```

出现频率高的词优先，最多保留 8 个关键词。

#### 为什么这样写

PPO reward 需要快速计算，不能每次都调用大模型。关键词奖励虽然粗糙，但稳定、便宜、可解释。面试时可以说这是第一版 rule-based accuracy reward，后续可替换为 BGE embedding 或 LLM judge。

### 关键源码：`convert_row()`

#### 源码

```python
def convert_row(row: dict[str, Any], row_id: int) -> dict[str, Any] | None:
    prompt = build_prompt(row)
    reference_answer = clean_text(row.get("output"))
    if not prompt or not reference_answer:
        return None

    keywords = candidate_terms(reference_answer)
    if not keywords:
        keywords = candidate_terms(prompt)
    if not keywords:
        return None

    risk_level = "high" if is_high_risk(prompt, reference_answer) else "normal"
    return {
        "id": f"medical-ppo-{row_id}",
        "prompt": prompt,
        "reference_answer": reference_answer,
        "answer_keywords": keywords,
        "risk_level": risk_level,
        "required_sections": DEFAULT_REQUIRED_SECTIONS,
        "source": row.get("source", "shibing624/medical"),
        "source_row_id": row.get("row_id", row_id),
        "similarity_score": row.get("similarity_score"),
        "best_target_id": row.get("best_target_id"),
    }
```

#### 这段代码在做什么

它把一条 SFT 数据转成一条 PPO reward 数据。

转换关系是：

```text
instruction + input -> prompt
output -> reference_answer
reference_answer -> answer_keywords
prompt + reference_answer -> risk_level
```

#### 逐段解释

第一段：

```python
prompt = build_prompt(row)
reference_answer = clean_text(row.get("output"))
if not prompt or not reference_answer:
    return None
```

如果问题或答案为空，这条样本不能用于 PPO 奖励训练，因为 reward 函数没有参考答案可用。

第二段：

```python
keywords = candidate_terms(reference_answer)
if not keywords:
    keywords = candidate_terms(prompt)
if not keywords:
    return None
```

优先从参考答案抽关键词。如果答案太短抽不出来，就从 prompt 里抽。如果还是没有关键词，就过滤掉。

第三段：

```python
risk_level = "high" if is_high_risk(prompt, reference_answer) else "normal"
```

根据高风险词判断样本是不是高风险病例。例如包含“胸痛”“心梗”“卒中”“出血”“呼吸困难”等词，就标成 `high`。

最后返回新的 PPO 记录，其中还保留：

- `similarity_score`
- `best_target_id`

这些来自 C-Eval 相似筛选阶段，方便回溯样本为什么进入 top100k。

### 关键源码：`build_dataset()`

#### 源码

```python
def build_dataset(input_path: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, row in enumerate(iter_jsonl(input_path)):
        record = convert_row(row, row_id)
        if record is not None:
            records.append(record)

    if len(records) < sample_size:
        raise ValueError(f"Only {len(records)} valid records found, need {sample_size}.")

    high_risk = [r for r in records if r["risk_level"] == "high"]
    normal = [r for r in records if r["risk_level"] != "high"]
    rng = random.Random(seed)
    rng.shuffle(high_risk)
    rng.shuffle(normal)

    min_high = min(len(high_risk), int(sample_size * 0.30))
    selected = high_risk[:min_high]
    selected.extend(normal[: sample_size - len(selected)])

    if len(selected) < sample_size:
        selected.extend(high_risk[min_high: min_high + sample_size - len(selected)])

    selected = selected[:sample_size]
    rng.shuffle(selected)
    return selected
```

#### 这段代码在做什么

它从 top100k SFT 数据里构建 5K PPO 数据，并保证其中至少约 30% 是高风险样本。

#### 逐段解释

第一段遍历 SFT 数据：

```python
for row_id, row in enumerate(iter_jsonl(input_path)):
    record = convert_row(row, row_id)
    if record is not None:
        records.append(record)
```

所有可转换样本都会先进入 `records`。

第二段做数量检查：

```python
if len(records) < sample_size:
    raise ValueError(...)
```

如果有效样本不足 5000，脚本直接报错，避免悄悄生成不完整数据。

第三段拆分高风险和普通样本：

```python
high_risk = [r for r in records if r["risk_level"] == "high"]
normal = [r for r in records if r["risk_level"] != "high"]
```

这样做是为了让安全奖励真的有训练信号。如果 PPO 数据里几乎没有高风险病例，`safety_score` 就学不到东西。

第四段抽样：

```python
min_high = min(len(high_risk), int(sample_size * 0.30))
selected = high_risk[:min_high]
selected.extend(normal[: sample_size - len(selected)])
```

默认 5000 条里至少取 1500 条高风险。最后再 shuffle，避免训练顺序先全是高风险再全是普通样本。

### 关键源码：`write_report()`

#### 源码

```python
def write_report(path: Path, rows: list[dict[str, Any]], input_path: Path) -> None:
    high_count = sum(1 for row in rows if row["risk_level"] == "high")
    report = {
        "input_path": str(input_path),
        "total_output": len(rows),
        "high_risk_count": high_count,
        "high_risk_ratio": round(high_count / len(rows), 4) if rows else 0,
        "required_sections": DEFAULT_REQUIRED_SECTIONS,
        "reward_weights": {"format": 0.30, "accuracy": 0.50, "safety": 0.20},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
```

#### 这段代码在做什么

写出 PPO 数据构建报告，记录：

- 输入文件
- 输出数量
- 高风险样本数量
- 高风险比例
- 格式要求
- reward 权重

当前正式报告里应看到：

```json
{
  "total_output": 5000,
  "high_risk_count": 1500,
  "high_risk_ratio": 0.3
}
```

## 奖励函数构建

PPO 训练入口是：

```text
MedicalGPT/training/ppo_medical_multireward.py
```

多维奖励权重在脚本顶部：

```python
FORMAT_WEIGHT = 0.30
ACCURACY_WEIGHT = 0.50
SAFETY_WEIGHT = 0.20
```

总奖励：

```text
total_reward =
  0.30 * format_score
+ 0.50 * accuracy_score
+ 0.20 * safety_score
```

### 关键源码：`compute_format_score()`

#### 源码

```python
def compute_format_score(response: str, required_sections: list[str] | None = None) -> float:
    response = normalize_text(response)
    sections = required_sections or DEFAULT_REQUIRED_SECTIONS
    if not response:
        return 0.0
    hits = 0
    for section in sections:
        section = normalize_text(section)
        if section and section in response:
            hits += 1
    return hits / max(len(sections), 1)
```

#### 这段代码在做什么

检查模型回答是否包含要求的四段：

```text
病情分析
处理建议
风险提示
就医建议
```

如果四段全有，格式分是 `1.0`；命中两段就是 `0.5`。

#### 为什么这样写

简历里说“复杂病例格式回答准确率由 72% 提升到 94%”，这个格式分就是优化这个指标的 reward signal。它不判断医学内容，只判断回答结构是否符合要求。

### 关键源码：准确率相关函数

#### 源码

```python
def compute_keyword_score(response: str, answer_keywords: list[str] | None = None) -> float:
    response = normalize_text(response).lower()
    keywords = [normalize_text(k).lower() for k in (answer_keywords or []) if normalize_text(k)]
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in response)
    return hits / len(keywords)
```

```python
def char_f1(prediction: str, reference: str) -> float:
    pred_chars = [c for c in normalize_text(prediction) if not c.isspace()]
    ref_chars = [c for c in normalize_text(reference) if not c.isspace()]
    if not pred_chars or not ref_chars:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for char in pred_chars:
        pred_counts[char] = pred_counts.get(char, 0) + 1
    for char in ref_chars:
        ref_counts[char] = ref_counts.get(char, 0) + 1
    overlap = sum(min(pred_counts.get(char, 0), count) for char, count in ref_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    return 2 * precision * recall / (precision + recall)
```

```python
def compute_accuracy_score(
    response: str,
    reference_answer: str | None = None,
    answer_keywords: list[str] | None = None,
) -> float:
    keyword_score = compute_keyword_score(response, answer_keywords)
    semantic_proxy = char_f1(response, reference_answer or "")
    if answer_keywords:
        return 0.7 * keyword_score + 0.3 * semantic_proxy
    return semantic_proxy
```

#### 这段代码在做什么

准确率分由两部分组成：

```text
关键词覆盖率：模型回答有没有覆盖标准答案里的医学要点
字符 F1：模型回答和参考答案的文本重合程度
```

最终：

```text
accuracy_score = 0.7 * keyword_score + 0.3 * char_f1
```

#### 逐段解释

`compute_keyword_score()` 逐个检查关键词是否出现在模型回答里。例如：

```text
answer_keywords = ["心电图", "肌钙蛋白", "急性冠脉综合征"]
```

如果模型回答命中两个，关键词分是：

```text
2 / 3 = 0.6667
```

`char_f1()` 是一个轻量语义相似代理。它不是真正 embedding，只是看字符集合重合程度。这样做的原因是 PPO 每一步都要算 reward，如果每次都跑 embedding 模型，训练会慢很多。

`compute_accuracy_score()` 让关键词覆盖占 70%，参考答案字符相似占 30%。医学问答里关键词更重要，因为模型不一定要逐字复述参考答案。

### 关键源码：`compute_safety_score()`

#### 源码

```python
def compute_safety_score(prompt: str, response: str, risk_level: str = "normal") -> float:
    prompt = normalize_text(prompt)
    response = normalize_text(response)
    text = f"{prompt}\n{response}"
    score = 1.0
    for term in SAFETY_NEGATIVE_TERMS:
        if term in response:
            score -= 0.25
    is_high = risk_level == "high" or any(term in prompt for term in HIGH_RISK_PROMPT_TERMS)
    if is_high and not any(term in response for term in SAFETY_POSITIVE_TERMS):
        score -= 0.35
    if any(term in response for term in SAFETY_POSITIVE_TERMS):
        score += 0.10
    if "诊断" in text and "医生" not in response and "检查" not in response:
        score -= 0.10
    return max(0.0, min(1.0, score))
```

#### 这段代码在做什么

安全分默认从 `1.0` 开始，然后根据危险表达扣分，根据安全提示加分。

危险表达包括：

```text
无需就医
不用就医
自行停药
自行用药
保证治愈
一定不是
不用检查
随便吃药
```

正向安全提示包括：

```text
及时就医
急诊
医生评估
医生指导
完善检查
不能替代医生诊断
面诊
```

#### 逐段解释

第一段：

```python
score = 1.0
for term in SAFETY_NEGATIVE_TERMS:
    if term in response:
        score -= 0.25
```

只要回答出现危险表达，就扣分。

第二段：

```python
is_high = risk_level == "high" or any(term in prompt for term in HIGH_RISK_PROMPT_TERMS)
if is_high and not any(term in response for term in SAFETY_POSITIVE_TERMS):
    score -= 0.35
```

如果是高风险病例，比如胸痛、卒中、呼吸困难，但回答没有建议及时就医或医生评估，就扣 0.35。

第三段：

```python
if any(term in response for term in SAFETY_POSITIVE_TERMS):
    score += 0.10
```

出现安全提示，会获得一点加分。

最后：

```python
return max(0.0, min(1.0, score))
```

把分数限制在 0 到 1 之间。

### 关键源码：`compute_total_reward()`

#### 源码

```python
def compute_total_reward(record: dict[str, Any], response: str) -> dict[str, float]:
    format_score = compute_format_score(response, record.get("required_sections"))
    accuracy_score = compute_accuracy_score(
        response,
        reference_answer=record.get("reference_answer"),
        answer_keywords=record.get("answer_keywords"),
    )
    safety_score = compute_safety_score(record.get("prompt", ""), response, record.get("risk_level", "normal"))
    total = FORMAT_WEIGHT * format_score + ACCURACY_WEIGHT * accuracy_score + SAFETY_WEIGHT * safety_score
    return {
        "format": float(format_score),
        "accuracy": float(accuracy_score),
        "safety": float(safety_score),
        "total": float(total),
    }
```

#### 这段代码在做什么

它把三个子分数组合成 PPO 需要的总 reward。

当前权重：

```text
format: 0.30
accuracy: 0.50
safety: 0.20
```

准确率权重最高，因为医学回答首先要答对。格式和安全作为强约束，避免模型只追求关键词而忽略回答规范。

## PPOTrainer 接口适配

TRL experimental PPOTrainer 需要 reward model 和 value model 都有特定接口。我们没有训练神经网络 reward model，所以要写 wrapper 来适配接口。

### 关键源码：`RuleBasedRewardModel`

#### 源码

```python
class RuleBasedRewardModel(torch.nn.Module):
    """TRL-compatible reward model wrapper backed by hand-written rules."""

    base_model_prefix = "backbone"

    def __init__(self, tokenizer: Any, prompt_index: list[tuple[str, str, dict[str, Any]]], pad_token_id: int):
        super().__init__()
        self.backbone = TokenPassthroughBackbone()
        self.score = RuleBasedScoreHead(tokenizer, prompt_index, pad_token_id)
        self.config = SimpleNamespace(model_type="rule_based_medical_reward")

    @property
    def last_batch_scores(self) -> list[dict[str, float]]:
        return self.score.last_batch_scores
```

#### 这段代码在做什么

它把手写奖励函数包装成 TRL 看起来像 reward model 的对象。

TRL 内部会调用：

```python
model.score(output.hidden_states[-1])
```

所以这个 wrapper 必须有：

- `base_model_prefix`
- `backbone`
- `score`
- `config`

但真正的奖励分数不是神经网络算出来的，而是 `RuleBasedScoreHead` decode 文本后调用 `compute_total_reward()`。

### 关键源码：`ValueScoreHead`

#### 源码

```python
class ValueScoreHead(torch.nn.Module):
    """A minimal trainable value head for TRL experimental PPO."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.proj.weight.device != hidden_states.device or self.proj.weight.dtype != hidden_states.dtype:
            self.proj.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return self.proj(hidden_states)
```

#### 这段代码在做什么

PPO 需要 value model 估计每个 token 的 value。`ValueScoreHead` 就是一个最小 value head：

```text
hidden_states -> Linear(hidden_size, 1) -> value
```

这里最重要的是 dtype 修复：

```python
self.proj.to(device=hidden_states.device, dtype=hidden_states.dtype)
```

因为 Qwen3 在 bfloat16 下输出的 `hidden_states` 是 `BFloat16`，而 PyTorch 线性层默认是 `Float32`。如果不转 dtype，会报：

```text
mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

### 关键源码：`ValueModelWrapper`

#### 源码

```python
class ValueModelWrapper(torch.nn.Module):
    """Wrap a causal LM with the attributes expected by TRL PPOTrainer.

    TRL experimental PPOTrainer expects value_model.base_model_prefix to point
    to the critic backbone, and get_reward(value_model, ...) expects the model
    to expose a .score(hidden_states) head.  Plain Qwen3ForCausalLM has neither
    .score nor a value head, so this wrapper adds only the value layer while
    keeping the original causal LM as the backbone.
    """

    base_model_prefix = "pretrained_model"

    def __init__(self, pretrained_model: torch.nn.Module):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = getattr(self.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size for PPO value head.")
        self.score = ValueScoreHead(hidden_size)

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return self.pretrained_model(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.pretrained_model, "gradient_checkpointing_enable"):
            return self.pretrained_model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.pretrained_model, "gradient_checkpointing_disable"):
            return self.pretrained_model.gradient_checkpointing_disable()
        return None
```

#### 这段代码在做什么

普通 `Qwen3ForCausalLM` 没有 `.score`，但 TRL PPO 的 value model 需要 `.score`。所以我们把 Qwen3 包一层：

```text
Qwen3ForCausalLM
  -> ValueModelWrapper
  -> .pretrained_model 是原模型
  -> .score 是新增 value head
```

`base_model_prefix = "pretrained_model"` 是给 TRL 用的。TRL 内部会执行类似：

```python
getattr(value_model, value_model.base_model_prefix)
```

如果没有这个字段，就会报：

```text
object has no attribute base_model_prefix
```

`forward()` 里强制：

```python
kwargs["output_hidden_states"] = True
kwargs["return_dict"] = True
```

因为 value head 需要最后一层 hidden states。

## 加载模型与构造 PPOTrainer

### 关键源码：`load_training_components()`

#### 源码

```python
def load_training_components(args: MedicalPPOArguments):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        from trl.experimental.ppo import PPOConfig, PPOTrainer
    except Exception:
        from trl import PPOConfig, PPOTrainer
    try:
        from peft import PeftModel
    except Exception:
        PeftModel = None

    torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
        "cache_dir": args.cache_dir,
        "device_map": "auto",
        "quantization_config": quantization_config,
    }
    policy = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    raw_value_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.peft_path:
        if PeftModel is None:
            raise ImportError("peft is required when --peft_path is used.")
        policy = PeftModel.from_pretrained(policy, args.peft_path, is_trainable=True)
        ref_model = PeftModel.from_pretrained(ref_model, args.peft_path, is_trainable=False)
        raw_value_model = PeftModel.from_pretrained(raw_value_model, args.peft_path, is_trainable=True)

    value_model = ValueModelWrapper(raw_value_model)

    records = iter_jsonl(args.train_file)
    rendered_prompts = [render_prompt(record["prompt"], args.template_name, tokenizer) for record in records]
    prompt_index = build_prompt_index(records, rendered_prompts)
    tokenized_rows = []
    for rendered_prompt in rendered_prompts:
        tokenized = tokenizer(
            rendered_prompt,
            truncation=True,
            max_length=args.max_prompt_length,
            padding=False,
            add_special_tokens=False,
        )
        tokenized_rows.append({"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]})
    dataset = Dataset.from_list(tokenized_rows)
    reward_model = RuleBasedRewardModel(tokenizer, prompt_index, tokenizer.pad_token_id)
    return PPOConfig, PPOTrainer, tokenizer, policy, ref_model, value_model, reward_model, dataset
```

#### 逐段解释

第一段延迟导入：

```python
from trl.experimental.ppo import PPOConfig, PPOTrainer
```

当前服务器使用的是 TRL experimental PPO。它 API 不稳定，所以脚本里做了 fallback。

第二段配置 4bit：

```python
quantization_config = BitsAndBytesConfig(...)
```

因为 PPO 要加载 policy、reference、value 三份模型，显存压力比 SFT 大很多，所以使用 4bit。

第三段加载 tokenizer：

```python
tokenizer.padding_side = "left"
```

decoder-only 模型生成时应该 left padding，否则 transformers 会警告：

```text
A decoder-only architecture is being used, but right-padding was detected
```

第四段加载三个模型：

```python
policy = AutoModelForCausalLM.from_pretrained(...)
ref_model = AutoModelForCausalLM.from_pretrained(...)
raw_value_model = AutoModelForCausalLM.from_pretrained(...)
```

三者职责不同：

- `policy` 会更新参数。
- `ref_model` 用于 KL 约束，不训练。
- `raw_value_model` 外面会包 `ValueModelWrapper`，用于 value 估计。

第五段处理 LoRA adapter：

```python
if args.peft_path:
    policy = PeftModel.from_pretrained(policy, args.peft_path, is_trainable=True)
    ref_model = PeftModel.from_pretrained(ref_model, args.peft_path, is_trainable=False)
    raw_value_model = PeftModel.from_pretrained(raw_value_model, args.peft_path, is_trainable=True)
```

如果输入的是合并模型，就不需要 `peft_path`。你当前使用的是：

```text
outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged
```

所以命令里不传 `--peft_path`。

第六段构造 reward model：

```python
reward_model = RuleBasedRewardModel(tokenizer, prompt_index, tokenizer.pad_token_id)
```

这不是训练出来的 reward model，而是接口包装器。真正分数来自规则函数。

### 关键源码：`parse_args()`

#### 源码

```python
def parse_args():
    from transformers import HfArgumentParser
    try:
        from trl.experimental.ppo import PPOConfig
    except Exception:
        from trl import PPOConfig

    # Compatibility with the command style used in the reproduction notes.
    # TRL's current PPOConfig uses num_ppo_epochs / kl_coef.
    argv = []
    for arg in sys.argv:
        if arg == "--ppo_epochs":
            argv.append("--num_ppo_epochs")
        elif arg == "--target_kl":
            argv.append("--kl_coef")
        else:
            argv.append(arg)
    sys.argv = argv

    parser = HfArgumentParser((MedicalPPOArguments, PPOConfig))
    medical_args, ppo_args = parser.parse_args_into_dataclasses()
    if getattr(ppo_args, "response_length", None) is not None:
        ppo_args.response_length = medical_args.max_new_tokens
    if getattr(ppo_args, "stop_token_id", None) is None:
        ppo_args.stop_token = "eos"
    return medical_args, ppo_args
```

#### 逐段解释

这段做了命令行兼容。

你习惯写：

```text
--ppo_epochs
--target_kl
```

但新版 TRL PPOConfig 里字段名是：

```text
--num_ppo_epochs
--kl_coef
```

所以脚本先把旧参数名替换成新参数名，避免命令报：

```text
Some specified arguments are not used
```

最后：

```python
ppo_args.response_length = medical_args.max_new_tokens
```

把我们自己的 `--max_new_tokens` 映射到 TRL PPO 生成长度。

### 关键源码：`main()`

#### 源码

```python
def main() -> None:
    medical_args, ppo_args = parse_args()
    PPOConfig, PPOTrainer, tokenizer, policy, ref_model, value_model, reward_model, dataset = load_training_components(
        medical_args
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
    )
    logger.info("*** Start medical PPO training with rule-based multi-reward ***")
    logger.info(f"Reward weights: format={FORMAT_WEIGHT}, accuracy={ACCURACY_WEIGHT}, safety={SAFETY_WEIGHT}")
    trainer.train()
    trainer.save_model(ppo_args.output_dir)
    tokenizer.save_pretrained(ppo_args.output_dir)
    logger.info(f"Saved PPO model to {ppo_args.output_dir}")
```

#### 逐段解释

`load_training_components()` 返回 PPO 需要的全部组件。

`PPOTrainer(...)` 里对应关系：

```text
model=policy
ref_model=reference model
reward_model=rule-based reward wrapper
value_model=Qwen3 + ValueScoreHead
train_dataset=PPO prompts
```

`trainer.train()` 会循环执行：

```text
从 dataset 取 prompt
policy 生成 response
reward_model 对 response 打分
value_model 估计 value
PPO 计算 policy loss / value loss / KL
更新 policy/value
```

## 调试记录

### 问题 1：`AutoModelForCausalLMWithValueHead` 不适配新版 TRL

旧写法：

```python
value_model = AutoModelForCausalLMWithValueHead.from_pretrained(...)
```

报错：

```text
AttributeError: 'AutoModelForCausalLMWithValueHead' object has no attribute 'base_model_prefix'
```

原因：

新版 TRL experimental PPOTrainer 初始化时会访问：

```python
value_model.base_model_prefix
```

旧 value head wrapper 没有这个字段。

解决：

```python
raw_value_model = AutoModelForCausalLM.from_pretrained(...)
value_model = ValueModelWrapper(raw_value_model)
```

### 问题 2：普通 Qwen3 缺少 `.score`

把 value model 改成普通 Qwen3 后，又报：

```text
'Qwen3ForCausalLM' object has no attribute 'score'
```

原因：

TRL 训练时调用：

```python
model.score(output.hidden_states[-1])
```

普通 causal LM 没有 `.score`。

解决：

```python
class ValueModelWrapper(torch.nn.Module):
    base_model_prefix = "pretrained_model"
    ...
    self.score = ValueScoreHead(hidden_size)
```

### 问题 3：BFloat16 和 Float dtype 不一致

报错：

```text
mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

原因：

Qwen3 hidden states 是 bfloat16，但新建的 `torch.nn.Linear` 默认是 float32。

解决：

```python
if self.proj.weight.device != hidden_states.device or self.proj.weight.dtype != hidden_states.dtype:
    self.proj.to(device=hidden_states.device, dtype=hidden_states.dtype)
```

### 问题 4：right padding 警告

警告：

```text
A decoder-only architecture is being used, but right-padding was detected
```

解决：

```python
tokenizer.padding_side = "left"
tokenizer.init_kwargs["padding_side"] = "left"
```

## 运行命令

### 构造 5K PPO 数据

```bash
python3 scripts/build_medical_ppo_dataset.py \
  --input data/sft/shibing624_medical_top100k.jsonl \
  --output data/rl/medical_complex_cases_5k.jsonl \
  --report data/rl/medical_complex_cases_5k_report.json \
  --sample-size 5000
```

验收：

```bash
wc -l data/rl/medical_complex_cases_5k.jsonl
cat data/rl/medical_complex_cases_5k_report.json
```

期望：

```text
5000
high_risk_ratio: 0.3
```

### 20 条 smoke test

```bash
cd /root/workspace/MedicalGPT

mkdir -p data/rl_smoke
head -n 20 data/rl/medical_complex_cases_5k.jsonl > data/rl_smoke/medical_complex_cases_20.jsonl

export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_PROJECT=MedSFT-Align
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRL_EXPERIMENTAL_SILENCE=1

rm -rf outputs/ppo_smoke_from_top100k_ckpt1000

CUDA_VISIBLE_DEVICES=0 python training/ppo_medical_multireward.py \
  --model_name_or_path outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged \
  --train_file data/rl_smoke/medical_complex_cases_20.jsonl \
  --output_dir outputs/ppo_smoke_from_top100k_ckpt1000 \
  --template_name qwen3 \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --trust_remote_code True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --max_prompt_length 512 \
  --max_new_tokens 128 \
  --ppo_epochs 1 \
  --target_kl 0.1 \
  --num_train_epochs 1 \
  --logging_strategy steps \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 2 \
  --report_to swanlab \
  --run_name ppo-smoke-20-from-top100k-ckpt1000
```

### 5K 正式训练

```bash
cd /root/workspace/MedicalGPT

export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_PROJECT=MedSFT-Align
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRL_EXPERIMENTAL_SILENCE=1

rm -rf outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000

CUDA_VISIBLE_DEVICES=0 python training/ppo_medical_multireward.py \
  --model_name_or_path outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged \
  --train_file data/rl/medical_complex_cases_5k.jsonl \
  --output_dir outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000 \
  --template_name qwen3 \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --trust_remote_code True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --max_prompt_length 768 \
  --max_new_tokens 256 \
  --ppo_epochs 1 \
  --target_kl 0.1 \
  --num_train_epochs 1 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 5 \
  --report_to swanlab \
  --run_name qwen3-4b-medical-ppo-multireward-5k
```

保存检查：

```bash
find outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000 -maxdepth 2 -type d | sort
```

## 面试解释口径

可以这样讲：

```text
PPO 阶段没有额外训练 reward model，而是设计了 rule-based multi-reward。
奖励函数由格式分、医学准确率分和安全分组成。
格式分约束复杂病例回答结构，准确率分用关键词覆盖率和参考答案字符 F1，安全分约束高风险病例必须提示就医并避免危险医疗建议。
为了接入 TRL PPOTrainer，我把规则奖励包装成 RuleBasedRewardModel，同时给 Qwen3 value model 加了 ValueScoreHead，使 PPO 具备 policy、reference、value、reward 四类组件。
```

如果面试官问“这算不算 PPO”：

```text
算 PPO，因为优化算法使用 PPOTrainer，reward signal 来自手写函数而不是神经网络 reward model。PPO 不要求 reward 必须来自模型，只要求能给采样回答提供标量奖励。
```

如果面试官问“为什么不用 Skywork reward model”：

```text
Skywork 是通用 reward model，医学格式和安全约束不够可控。我把它作为可选 baseline，主实验使用可解释的多维规则奖励，便于针对复杂病例格式、医学要点和安全提示做定向优化。
```

## 常见坑

- PPO 数据不能直接用 SFT JSONL，必须有 `reference_answer / answer_keywords / risk_level`。
- 只用 Skywork reward model 不能体现“格式分 + 准确率分 + 安全分”的手写多维奖励设计。
- TRL experimental PPO API 不稳定，`AutoModelForCausalLMWithValueHead` 可能不适配新版。
- value model 必须有 `.score`。
- value head dtype 必须和 hidden states 一致。
- decoder-only 模型生成要 left padding。
- PPO 会加载 policy、reference、value 三份模型，显存压力明显大于 SFT。
- 先跑 20 条 smoke test，再跑 5K。

## 学习检查清单

- 能说清楚为什么 SFT-100k 要派生 PPO 5K 数据。
- 能解释 `prompt / reference_answer / answer_keywords / risk_level / required_sections` 每个字段的作用。
- 能手写格式分、准确率分、安全分的计算逻辑。
- 能解释为什么这里没有训练 reward model，但仍然有 reward signal。
- 能说清楚 PPO 的 policy、reference、value、reward 四个组件。
- 能解释 `RuleBasedRewardModel` 和 `ValueModelWrapper` 为什么存在。
- 能复盘三个关键 bug：value head API、`.score`、dtype mismatch。
