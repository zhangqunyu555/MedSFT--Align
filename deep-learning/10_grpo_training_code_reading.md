# MedicalGPT 训练源码精读 10：grpo_training.py

## 整体作用

GRPO 是一种基于奖励函数的强化学习训练方式。MedicalGPT 的 `grpo_training.py` 示例更偏数学 / 格式奖励，但它对你后续做医疗复杂病例格式奖励很有启发。

它的核心思路是：

```text
模型生成多个回答
  -> 奖励函数打分
  -> 根据奖励优化模型
```

## 参数类：`ScriptArguments`

```python
class ScriptArguments:
    tokenizer_name_or_path: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default="openai/gsm8k")
    train_file_dir: Optional[str] = field(default=None)
    train_samples: Optional[int] = field(default=-1)
    subset_name: Optional[str] = field(default="main")
    dataset_splits: Optional[str] = field(default="train")
    preprocessing_num_workers: Optional[int] = field(default=10)
    qlora: bool = field(default=False)
```

解释：

- 默认数据集是 GSM8K。
- 也支持本地 `train_file_dir`。
- `qlora` 控制是否用 QLoRA。

## 文本标准化

```python
def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text
```

解释：

- 空值返回空字符串。
- 多个空白压成一个空格。
- 转小写。

奖励函数比较答案时，经常需要先做标准化。

## 提取答案

```python
def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
```

解释：

它从模型输出里提取 `<answer>...</answer>` 中间的内容。

如果模型输出：

```text
<think>推理过程</think><answer>42</answer>
```

提取结果是：

```text
42
```

## 准确率奖励：`accuracy_reward()`

```python
def accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        if '####' in sol:
            gold_parsed = parse(sol.split("####", 1)[-1].strip())
            answer_parsed = parse(extract_answer(content))
        else:
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            logger.warning(f"Error in verification: {e}")
            reward = 0.0
        rewards.append(reward)
    return rewards
```

逐段解释：

- `completions` 是模型生成结果。
- `answer` 是标准答案。
- 如果标准答案里有 `####`，按 GSM8K 格式提取最终答案。
- 否则尝试用 latex parser 解析。
- `verify(answer_parsed, gold_parsed)` 判断预测答案和标准答案是否等价。
- 正确奖励 1.0，错误奖励 0.0。

## 格式奖励：`format_reward()`

```python
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards
```

解释：

这个函数只检查输出格式：

```text
<think>...</think><answer>...</answer>
```

格式对，奖励 1；格式不对，奖励 0。

## 系统提示

```python
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. ..."
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags..."
)
```

解释：

它要求模型按固定格式输出思考和答案。

后续医疗复杂病例也可以设计类似格式：

```text
<analysis>病情分析</analysis>
<diagnosis>可能诊断</diagnosis>
<advice>就医建议</advice>
<safety>安全提醒</safety>
```

## 数据准备逻辑

```python
dataset = dataset.map(
    lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['answer']
    },
    num_proc=script_args.preprocessing_num_workers,
)
```

解释：

GRPO 数据需要：

- `prompt`：给模型生成
- `answer`：给奖励函数校验

这和 SFT 的 `labels` 不一样。

## 和当前医疗项目的关系

GRPO 很适合你后续的复杂病例格式优化：

- 格式奖励：是否按病例回答模板输出。
- 准确率奖励：答案是否和参考诊断/检查建议一致。
- 安全奖励：是否包含“不能替代医生诊断”等安全提示。

## 常见坑

- 奖励函数写得不好，模型会 reward hacking。
- 格式奖励太强，模型可能只学格式不学医学内容。
- 医疗准确率奖励比数学更难，需要可靠参考答案或强模型裁判。
- GRPO 前最好先有一个 SFT 模型。

