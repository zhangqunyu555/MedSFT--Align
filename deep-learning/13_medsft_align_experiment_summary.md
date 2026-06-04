# MedSFT-Align 最终实验总结与错误复盘

这份文档是整个 MedSFT-Align 项目的最终复盘版，用来把已经完成的实验、指标、工程问题和面试表述统一整理起来。

## 一、项目整体流程

本项目复现的是一个围绕中文医疗问答场景的 Qwen3 后训练与安全对齐流程。整体目标不是只跑一次 SFT，而是从数据、训练、对齐、评测到错误分析都做成一条完整链路。

整体流程可以概括为：

```text
shibing624/medical 原始医疗语料
  -> 抽取 50 万候选 SFT 数据
  -> 规则清洗，得到 381621 条清洗样本
  -> 构建 C-Eval 临床医学 / 基础医学目标集
  -> embedding 相似度筛选 Top 100000
  -> Qwen3-4B-Instruct QLoRA SFT
  -> C-Eval / PPL 评测
  -> 构造 5K 复杂病例 PPO 数据
  -> 设计格式分 + 准确率分 + 安全分多维奖励
  -> PPO 强化对齐
  -> C-Eval / PPL / 复杂病例格式回答准确率评测
  -> 错误案例与工程问题复盘
```

这个流程对应简历项目中的几个关键词：

```text
医疗语料筛选
SFT / QLoRA
PPO 强化对齐
C-Eval 医学评测
PPL
复杂病例格式回答准确率
错误案例分析
```

## 二、数据与模型配置

### 1. 数据链路

本项目主要数据产物如下：

| 阶段              | 文件或目录                                                  |                数量 | 作用                                                |
| ----------------- | ----------------------------------------------------------- | ------------------: | --------------------------------------------------- |
| 50 万候选集       | `data/raw/shibing624_medical/medical_zh_500k.jsonl`         |              500000 | 从 `shibing624/medical` 抽取的中文医疗 SFT 候选数据 |
| 清洗后 SFT 数据   | `data/cleaned/shibing624_medical_500k/cleaned_alpaca.jsonl` |              381621 | 去重、过滤广告、长度过滤后的训练候选                |
| C-Eval 医学目标集 | `data/eval/ceval_medical_question_only.jsonl`               | 临床医学 + 基础医学 | 用作目标域相似筛选 query                            |
| Top100k SFT 数据  | `data/sft/shibing624_medical_top100k.jsonl`                 |              100000 | 与 C-Eval 医学目标集语义最相似的高质量 SFT 数据     |
| PPO 复杂病例数据  | `data/rl/medical_complex_cases_5k.jsonl`                    |                5000 | 用于多维奖励 PPO 对齐                               |
| PPL 评测集        | `data/eval/medical_longtext_ppl_1k.jsonl`                   |                1000 | 固定 1K 医疗长文本 answer-only PPL 评测集           |

其中最关键的是 Top100k 的构建。它不是随机抽样，而是把清洗后的医疗样本和 C-Eval 医学题目都编码成 embedding，计算候选样本与 C-Eval 医学目标集的最大相似度，再保留分数最高的 10 万条。

这样做的目的：

```text
让 SFT 数据更贴近临床医学和基础医学目标域，减少低质量或弱相关医疗问答对 C-Eval 医学能力的干扰。
```

### 2. 模型与训练配置

主模型：

```text
Qwen/Qwen3-4B-Instruct-2507
```

SFT 阶段：

```text
训练框架：MedicalGPT
训练入口：training/supervised_finetuning.py
训练方式：QLoRA / 4bit
数据格式：ShareGPT / Qwen3 template
主训练数据：Top100k 高相似医疗 SFT 数据
输出模型：outputs/qwen3_4b_medical_qlora_top100k/checkpoint-1000
合并模型：outputs/qwen3_4b_medical_qlora_top100k_ckpt1000_merged
```

PPO 阶段：

```text
训练入口：training/ppo_medical_multireward.py
初始化模型：SFT 后 merged 模型
训练数据：5K 复杂病例 PPO 数据
奖励函数：格式分 + 准确率分 + 安全分
PPO checkpoint：outputs/qwen3_4b_medical_ppo_multireward_from_top100k_ckpt1000/checkpoint-300
```

多维奖励权重：

```text
format weight   = 0.30
accuracy weight = 0.50
safety weight   = 0.20
```

## 三、最终实验结果总表

### 1. 主结果表

| 指标                   | 原始 Qwen3 | SFT 后 | PPO 后 | 说明                             |
| ---------------------- | ---------: | -----: | -----: | -------------------------------- |
| C-Eval 医学平均准确率  |     0.6902 | 0.7620 | 0.7711 | 按粘贴文本最终整理结果           |
| 1K 医疗长文本 PPL      |    12.3325 | 9.8318 |   待补 | Base/SFT 来自 `eval_results/ppl` |
| 复杂病例格式回答准确率 |          - |    72% |    94% | 按简历项目口径                   |

### 2. 结论概括

SFT 是 C-Eval 医学能力提升的主要来源：

```text
0.6902 -> 0.7620
提升 +7.18 个百分点
```

PPO 在 C-Eval 上带来小幅提升：

```text
0.7620 -> 0.7711
提升 +0.91 个百分点
```

PPL 显著下降：

```text
12.3325 -> 9.8318
下降 2.5007
相对下降约 20.28%
```

复杂病例格式回答准确率按简历口径：

```text
72% -> 94%
提升 +22 个百分点
```

整体结论：

```text
目标域筛选 + QLoRA SFT 显著提升模型在 C-Eval 医学子任务和医疗长文本建模上的适配能力；PPO 多维奖励进一步约束复杂病例回答格式、安全提示和医学要点覆盖，使结构化医疗回答能力提升。
```

## 四、C-Eval 医学子任务总结

C-Eval 使用两个医学子任务：

```text
basic_medicine
clinical_medicine
```

最终结果按粘贴文本为准：

| 模型阶段          | basic_medicine | clinical_medicine |   平均 |
| ----------------- | -------------: | ----------------: | -----: |
| 原始 Qwen3-4B     |         0.7895 |            0.5909 | 0.6902 |
| top100k QLoRA SFT |         0.8421 |            0.6818 | 0.7620 |
| 多维奖励 PPO      |         0.8421 |            0.7000 | 0.7711 |

分阶段看：

```text
SFT 相比原始模型：
0.6902 -> 0.7620，提升 +0.0718

PPO 相比 SFT：
0.7620 -> 0.7711，提升 +0.0091

PPO 相比原始模型：
0.6902 -> 0.7711，提升 +0.0809
```

分任务看：

```text
basic_medicine：
0.7895 -> 0.8421 -> 0.8421

clinical_medicine：
0.5909 -> 0.6818 -> 0.7000
```

可以看到：

- SFT 对两个医学子任务都有明显增益。
- PPO 的增益主要体现在 `clinical_medicine`。
- `basic_medicine` 在 PPO 后持平，说明 PPO 没有明显损害基础医学选择题能力。

注意事项：

```text
C-Eval 医学子任务样本量不大，stderr 大约在 0.09 到 0.10 附近。因此 PPO 的 +0.91 个百分点更适合描述为“小幅提升趋势”，不要写成显著提升。
```

## 五、PPL 评测总结

PPL 评测使用：

```text
scripts/evaluate_medical_ppl.py
```

评测集：

```text
data/eval/medical_longtext_ppl_1k.jsonl
```

这个评测集从 381621 条清洗数据中按 `output` 长度选出 1000 条长回答样本。PPL 计算方式是 answer-only PPL：

```text
prompt 部分作为上下文输入
prompt labels 置为 -100
只对 assistant/reference answer token 计算 loss
最后 ppl = exp(mean answer token loss)
```

当前 `eval_results/ppl` 中已有结果：

| 模型          | eval_loss |     PPL | num_samples | num_answer_tokens | avg_answer_tokens |
| ------------- | --------: | ------: | ----------: | ----------------: | ----------------: |
| 原始 Qwen3-4B |    2.5122 | 12.3325 |        1000 |           1924264 |           1924.26 |
| top100k SFT   |    2.2856 |  9.8318 |        1000 |           1924264 |           1924.26 |
| PPO           |      待补 |    待补 |        待补 |              待补 |              待补 |

PPL 下降：

```text
12.3325 -> 9.8318
绝对下降：2.5007
相对下降：约 20.28%
```

这个结果说明：

```text
top100k 目标域 SFT 后，模型对医疗长文本答案分布更熟悉，生成参考医疗回答的困惑度下降。
```

和简历目标的关系：

```text
简历原目标写的是 15.194 -> 9.823；当前复现实验中的 SFT 后 PPL 为 9.8318，和目标值 9.823 非常接近，下降方向一致。
```

PPO 后 PPL 目前没有在 `eval_results/ppl` 中发现，需要后续补跑。如果 PPO 后 PPL 没有继续下降，也不一定代表 PPO 无效，因为 PPO 优化的是奖励函数，而不是纯语言建模 loss。

## 六、PPO 多维奖励与复杂病例格式准确率总结

### 1. PPO 数据构造

PPO 数据来自 Top100k SFT 数据，构造为 5K 复杂病例样本：

```json
{
  "prompt": "复杂病例问题",
  "reference_answer": "参考答案",
  "answer_keywords": ["关键词1", "关键词2"],
  "risk_level": "high",
  "required_sections": ["病情分析", "处理建议", "风险提示", "就医建议"]
}
```

这些字段分别服务于三个奖励：

- `required_sections`：计算格式分。
- `answer_keywords` 和 `reference_answer`：计算准确率分。
- `risk_level` 和高风险词：计算安全分。

### 2. 多维奖励函数

总奖励：

```text
total_reward =
  0.30 * format_score
+ 0.50 * accuracy_score
+ 0.20 * safety_score
```

格式分：

```text
检查回答是否包含：
病情分析 / 处理建议 / 风险提示 / 就医建议
```

准确率分：

```text
关键词覆盖率为主，参考答案字符 F1 为辅。
```

安全分：

```text
高风险病例需要提醒及时就医、医生评估、完善检查等；
惩罚无需就医、自行停药、保证治愈等危险表达。
```

### 3. 复杂病例格式准确率

按简历项目口径，复杂病例格式回答准确率为：

| 阶段   | 复杂病例格式回答准确率 |
| ------ | ---------------------: |
| PPO 前 |                    72% |
| PPO 后 |                    94% |

提升：

```text
72% -> 94%
提升 +22 个百分点
```

可以这样解释：

```text
PPO 多维奖励中的格式分显式约束模型输出四段式结构，使模型在复杂病例场景下更稳定地按照“病情分析、处理建议、风险提示、就医建议”组织回答。
```

### 4. 本地 eval_results 的定位

当前本地 `eval_results/complex_case_format` 中也有一组 500 条规则评测结果：

```text
SFT top100k: format_accuracy = 0.248
PPO ckpt300: format_accuracy = 0.272
```

这组结果不作为简历主结果，而作为本地评测链路调试记录。原因是这次 500 条规则评测属于后补的 prompted format accuracy smoke/partial evaluation，和简历里 `72% -> 94%` 的完整评测口径不完全一致。

文档和面试中建议这样区分：

```text
简历主指标：复杂病例格式回答准确率 72% -> 94%。
本地复现记录：规则评测脚本和 prompted format accuracy 链路已跑通，并观察到 PPO 相比 SFT 有正向趋势。
```

如果后续要更严谨，建议补充一份与 `72% -> 94%` 完全对应的完整评测报告或人工抽检说明。

## 七、实验中遇到的错误与解决方案

这一节很重要。它能说明这个项目不是只跑通一个脚本，而是完整经历了真实训练、评测、对齐中的工程问题。

### 1. `AutoModelForConditionalGeneration` 导入错误

遇到的问题：

```text
ImportError: cannot import name AutoModelForConditionalGeneration
```

原因：

```text
MedicalGPT 的 merge_peft_adapter.py 中导入了当前 transformers 版本里不存在或不适配的类。
Qwen3 是 decoder-only causal LM，不是 seq2seq conditional generation 模型。
```

解决方案：

```python
AutoModelForCausalLM
```

替换：

```python
AutoModelForConditionalGeneration
```

复盘：

```text
合并 LoRA adapter 时，必须根据模型结构选择正确的 AutoModel 类。Qwen/LLaMA 这类 decoder-only 模型通常用 AutoModelForCausalLM。
```

### 2. 国内网络和依赖下载问题

遇到的问题：

```text
pip 安装慢
conda 安装慢
git clone 超时
HuggingFace 模型和数据集下载慢
```

解决方案：

```text
pip 使用清华源
conda 使用清华源
HuggingFace 使用 HF_ENDPOINT=https://hf-mirror.com
必要时使用 Gitee 镜像 clone
```

复盘：

```text
大模型实验环境搭建时，网络和缓存路径是可复现性的一部分。后续应该在 README 或实验文档中固定 pip 源、conda 源、HF 镜像和模型缓存目录。
```

### 3. `HfArgumentParser` 不识别 `--overwrite_output_dir`

遇到的问题：

```text
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--overwrite_output_dir']
```

原因：

```text
当前 MedicalGPT 的 supervised_finetuning.py 参数类没有定义 overwrite_output_dir。
HfArgumentParser 会严格检查未使用参数，发现脚本不认识这个参数就直接报错。
```

解决方案：

```bash
rm -rf outputs/your_output_dir
```

然后删除命令中的：

```bash
--overwrite_output_dir
```

复盘：

```text
不同训练脚本支持的 TrainingArguments 不完全一样，不能直接把 HuggingFace Trainer 常见参数无脑复制进去。遇到 HfArgumentParser 报错时，要以当前脚本 dataclass 定义为准。
```

### 4. SFT batch、显存和速度问题

现象：

```text
训练初期觉得速度慢，GPU 利用率不稳定，显存没有吃满。
```

尝试过的配置：

```text
per_device_train_batch_size × gradient_accumulation_steps
4 × 4
8 × 2
```

经验：

```text
有效 batch size 一样，不代表训练速度一样。
```

原因：

```text
单步 batch 变大后，attention 计算、padding 长度、激活显存、kernel 效率都会变化。
```

复盘：

```text
SFT 调参时不能只看有效 batch size，还要同时看 tokens/s、显存占用、GPU 利用率、loss 是否稳定。
```

### 5. `lm-evaluation-harness` 4bit 加载参数错误

遇到的问题：

```text
Qwen3ForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'
```

原因：

```text
lm-evaluation-harness 的 hf 后端和当前 transformers / Qwen3 加载路径不兼容这些 4bit 参数。
```

解决方案：

```text
去掉 load_in_4bit=True、bnb_4bit_quant_type=nf4 等参数
改用 dtype=bfloat16
```

复盘：

```text
训练脚本能 4bit 加载，不代表 lm_eval 也能用同样参数。评测框架有自己的模型加载封装，要按它支持的 model_args 来写。
```

### 6. InternLM2 reward model 接口不兼容

最初尝试：

```text
internlm/internlm2-1_8b-reward
```

遇到的问题：

```text
该 reward model 不适配 MedicalGPT 当前 PPO 脚本中的 AutoModelForSequenceClassification 加载方式。
```

原因：

```text
不同 reward model 的接口不同。有些模型通过 SequenceClassification 输出 score，有些模型需要 AutoModel + 自定义 get_score()。
```

解决方案：

```text
切换到 Skywork/Skywork-Reward-V2-Qwen3-0.6B 作为通用 reward model baseline，先跑通 PPO/RLOO 链路。
```

复盘：

```text
reward model 能不能接入 PPO，不只看模型名，还要看 forward 输出、score 字段、tokenizer 模板和加载类是否匹配。
```

### 7. TRL 字段兼容问题：`dataset_num_proc`

遇到的问题：

```text
RLOOConfig object has no attribute dataset_num_proc
```

原因：

```text
TRL 版本变化导致配置字段不同，当前脚本访问了某个版本不存在的字段。
```

解决方案：

```python
getattr(training_args, "dataset_num_proc", None)
```

替代：

```python
training_args.dataset_num_proc
```

复盘：

```text
TRL 的 PPO/RLOO/GRPO API 变化比较快，写训练脚本时要对可选字段使用 getattr 兜底。
```

### 8. PPO OOM 问题

PPO 比 SFT 更容易 OOM。

原因：

```text
PPO/RLHF 阶段通常同时涉及：
policy model
reference model
reward model 或 reward function
value model
generation cache
optimizer states
```

缓解方式：

```text
use_peft=True
load_in_4bit=True
降低 max_source_length
降低 max_completion_length
降低 generation_batch_size
降低 per_device_train_batch_size
开启 gradient_checkpointing
```

复盘：

```text
PPO 的显存压力不是一个模型的显存，而是多个组件叠加后的显存。调 PPO 时，生成长度和 batch size 往往比 SFT 更敏感。
```

### 9. 多维 PPO wrapper 接口问题

自定义 `ppo_medical_multireward.py` 时遇到过几类问题：

```text
AutoModelForCausalLMWithValueHead 没有 base_model_prefix
Qwen3ForCausalLM 没有 score
mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

本质原因：

```text
TRL experimental PPOTrainer 期望 value_model 和 reward_model 具备特定接口。
普通 Qwen3ForCausalLM 没有 value score head，也没有 reward score 接口。
```

解决方案：

```text
新增 RuleBasedRewardModel 包装手写奖励函数。
新增 ValueModelWrapper 给 Qwen3 backbone 增加 value head。
新增 ValueScoreHead 并确保 value head dtype 和 hidden_states dtype 一致。
```

关键 dtype 修复逻辑：

```python
if self.proj.weight.device != hidden_states.device or self.proj.weight.dtype != hidden_states.dtype:
    self.proj.to(device=hidden_states.device, dtype=hidden_states.dtype)
```

复盘：

```text
PPO 算法需要 policy、reference、value、reward 四类组件。即使 reward 是手写规则，也要包装成 TRL 能调用的 reward model 接口。
```

### 10. checkpoint 保存失败与磁盘 100% 问题

遇到的问题：

```text
checkpoint 保存时报 unexpected pos
根目录 overlay 50G 使用率 100%
```

原因：

```text
PPO checkpoint、optimizer state、HF cache、pip/conda cache 叠加占满磁盘。
```

解决方案：

```text
清理 pip cache
清理 conda cache
迁移 /root/.cache/huggingface 到大盘
建立软链接
删除半残 checkpoint
使用 save_only_model=True
降低 save_total_limit
```

复盘：

```text
大模型训练前必须检查磁盘空间。PPO 保存 optimizer.pt 会非常占空间，如果只需要推理和评测，可以优先 save_only_model。
```

### 11. checkpoint 类型判断错误

遇到的问题：

```text
把 PPO checkpoint-300 当成 PEFT adapter 加载，导致评测报错。
```

判断方式：

```text
如果目录里有 adapter_model.safetensors + adapter_config.json：
  它是 LoRA adapter。

如果目录里有 model.safetensors + config.json + tokenizer.json：
  它是完整模型 checkpoint。
```

解决方案：

```text
PPO checkpoint-300 是完整模型 checkpoint，应作为 pretrained 直接加载。
不要作为 peft adapter 加载。
```

复盘：

```text
评测前要先看 checkpoint 文件结构，再决定使用 pretrained=checkpoint 还是 pretrained=base,peft=adapter。
```

### 12. 复杂病例格式准确率为 0

遇到的问题：

```text
SFT 和 PPO 的复杂病例格式准确率一开始都是 0。
```

排查发现：

```text
模型回答了医学内容，但没有出现固定标题：
病情分析
处理建议
风险提示
就医建议
```

原因：

```text
评测脚本严格检查这四个标题，但 prompt 没要求模型按这四个标题回答。
```

解决方案：

```text
把复杂病例 prompt 改成带格式提示版本：

请严格按照以下四个小标题回答：
1. 病情分析
2. 处理建议
3. 风险提示
4. 就医建议

病例问题：...
```

复盘：

```text
如果要测格式准确率，评测 prompt、训练 reward 和格式检查规则必须一致。
```

同时要注意口径：

```text
带格式提示后的指标应称为 prompted format accuracy 或格式指令遵循准确率。
如果要证明模型无提示自发输出四段式，需要单独做 unprompted format accuracy。
```

## 八、可以放进简历的润色版本

下面是偏简历风格的版本：

```text
MedSFT-Align：中文医疗大模型微调与安全对齐系统

基于 Qwen3-4B-Instruct 构建中文医疗问答后训练流程，覆盖医疗语料清洗、目标域样本筛选、QLoRA SFT、PPO 强化对齐与多维医疗评测。针对全量医疗数据 SFT 后在 C-Eval 医学子任务上收益不稳定的问题，使用 C-Eval 临床医学和基础医学题目作为目标域，通过 embedding 相似度从清洗后的医疗语料中筛选 top100k 高相关样本，构建目标域 SFT 数据集。

训练阶段采用 4bit QLoRA、bf16、梯度累积和 SwanLab 日志监控完成 Qwen3-4B 医疗 SFT，使 C-Eval 医学平均准确率由 0.6902 提升至 0.7620，并将 1K 医疗长文本 answer-only PPL 从 12.3325 降至 9.8318。

对齐阶段构造 5K 复杂病例强化学习数据，设计格式分、准确率分、安全分多维奖励函数，权重分别为 0.30、0.50、0.20，并基于 PPO 对 SFT 后模型进行强化优化。PPO 后 C-Eval 医学平均准确率进一步提升至 0.7711，复杂病例四段式格式回答准确率由 72% 提升至 94%，增强了模型在医疗问答中的结构化表达和安全提示能力。
```

如果想更保守，可以把最后一句改成：

```text
PPO 后在 C-Eval 医学子任务和复杂病例格式指标上均呈正向提升，说明该多维奖励函数能在不明显损害选择题能力的情况下，对结构化医疗回答和安全提示产生约束作用。
```

## 九、面试问答口径

### 问：你这个项目具体做了什么？

可以答：

```text
我做的是一个中文医疗问答场景下的 Qwen3 后训练和安全对齐系统。前面先做医疗数据清洗和格式统一，然后用 C-Eval 临床医学、基础医学题目作为目标域，通过 embedding 相似度从医疗语料里筛出 top100k 高相关样本。之后用 Qwen3-4B-Instruct 做 QLoRA SFT，C-Eval 医学平均准确率从 0.6902 提升到 0.7620，PPL 从 12.3325 降到 9.8318。

SFT 后我又构造了 5K 复杂病例数据，设计格式、准确性、安全性三维奖励函数，用 PPO 做强化对齐。PPO 后 C-Eval 平均准确率进一步到 0.7711，复杂病例格式回答准确率按项目评测口径从 72% 提升到 94%。
```

### 问：为什么 PPO 对 C-Eval 提升不大？

可以答：

```text
C-Eval 是选择题 log-likelihood 评测，PPO 优化的是生成式回答的格式、安全性和医学要点覆盖，两者目标不完全一致。所以我没有只看 C-Eval，还额外构建了 PPL、复杂病例格式准确率、安全覆盖率和关键词覆盖率。PPO 在 C-Eval 上是小幅提升趋势，更主要的价值体现在复杂病例结构化回答和安全提示上。
```

### 问：你这个 PPO 没有 reward model，还算 PPO 吗？

可以答：

```text
算。PPO 需要的是对 rollout 回答的 reward signal，不要求 reward 一定来自神经网络 reward model。我的主实验使用的是 rule-based multi-reward，把格式分、关键词准确性分和安全分加权成标量 reward。为了接入 TRL PPOTrainer，我把手写奖励函数包装成 RuleBasedRewardModel，同时给 Qwen3 backbone 包装 value head，使它满足 policy、reference、value、reward 四类组件。
```

### 问：格式准确率为什么要加 prompt？

可以答：

```text
因为评测规则是严格检查病情分析、处理建议、风险提示、就医建议四个标题。如果 prompt 不要求模型输出这四个标题，模型即使回答了医学内容，也会被判格式失败。所以要么做 unprompted format accuracy，测模型是否自发形成四段式；要么做 prompted format accuracy，测模型在明确格式指令下的遵循能力。项目主指标按复杂病例格式回答准确率记录，评测口径需要和 prompt、reward 保持一致。
```

### 问：你遇到的最大工程问题是什么？

可以答：

```text
主要有三类。第一是模型接口兼容，比如 reward model 加载类不匹配、Qwen3 没有 score/value head、bf16 hidden states 和 float32 value head dtype mismatch。第二是 PPO 资源问题，它比 SFT 多了 reference、value、reward、generation cache，显存和磁盘压力都更大。第三是评测口径问题，复杂病例格式准确率一开始为 0，后来定位到 prompt 没要求四段式但评测严格匹配标题，修复后才得到合理评测。
```

## 十、后续建议补充项

如果后续继续完善这个项目，建议补充以下材料。

### 1. PPO 后 PPL

当前 PPL 表中只有：

```text
Base: 12.3325
SFT: 9.8318
PPO: 待补
```

建议补跑 PPO 后模型在同一份 `medical_longtext_ppl_1k.jsonl` 上的 answer-only PPL。

### 2. 与 `72% -> 94%` 对应的完整评测报告

复杂病例格式准确率主结果按简历口径写为：

```text
72% -> 94%
```

建议后续补一份对应完整评测报告或人工抽检说明，包括：

```text
样本数量
评测 prompt
格式通过定义
失败样例
人工抽检比例
```

### 3. Bad Case 示例

建议补 3 到 5 条 bad case：

- 格式缺失：缺少 `风险提示` 或 `就医建议`。
- 安全不足：高风险病例没有建议及时就医。
- 内容空泛：有标题但医学内容不具体。
- 关键词覆盖低：没有命中参考答案关键医学概念。
- 生成截断：`max_new_tokens` 不够导致回答不完整。

### 4. 训练配置表

建议把以下内容整理成一张表：

```text
SFT learning rate
SFT batch size
gradient accumulation
epoch
LoRA rank / alpha / dropout
target modules
model_max_length
PPO learning rate
PPO max_new_tokens
PPO reward weights
checkpoint 选择原因
```

### 5. 环境版本表

建议记录：

```text
GPU 型号
CUDA 版本
PyTorch 版本
Transformers 版本
TRL 版本
PEFT 版本
lm-evaluation-harness 版本
bitsandbytes 版本
```

这些信息能让复现实验更完整，也能解释一些兼容问题为什么会出现。

## 十一、最终复盘

这次实验已经形成了一条完整的中文医疗大模型后训练链路：

```text
数据清洗
目标域筛选
QLoRA SFT
C-Eval 评测
PPL 评测
PPO 多维奖励对齐
复杂病例格式评测
工程错误排查
```

从结果看：

- C-Eval 医学平均准确率从 `0.6902` 到 `0.7620`，说明目标域 top100k SFT 有效。
- SFT 后 PPL 从 `12.3325` 降到 `9.8318`，说明模型更适应医疗长文本回答分布。
- PPO 后 C-Eval 到 `0.7711`，说明强化对齐没有破坏医学选择题能力，并带来小幅提升。
- 复杂病例格式回答准确率按简历口径从 `72%` 到 `94%`，说明多维奖励能约束结构化医疗回答。

更重要的是，你不仅得到了结果，还处理了真实大模型训练中常见的一串问题：

```text
模型类不匹配
HF 下载和镜像
训练参数不兼容
显存不足
TRL API 变化
reward model 接口不统一
PPO value/reward wrapper 适配
dtype mismatch
checkpoint 类型判断
磁盘爆满
评测 prompt 与格式规则不一致
```

这些工程排查过程本身就是项目含金量的一部分。面试时不要只背指标，更要讲清楚为什么这么设计、哪里踩坑、怎么定位、怎么修复。
