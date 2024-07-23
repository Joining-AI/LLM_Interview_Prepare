[![GitHub stars](https://img.shields.io/github.com/Joining-AI/LLM_Interview_Prepare?style=social)](https://github.com/Joining-AI/LLM_Interview_Prepare)

### 2.1 大模型 基础面

  2.1.1 目前主流的开源模型体系有哪些?

  2.1.2 prefix LM 和 causal LM 区别是什么?

  2.1.3 涌现能力是啥原因?

  2.1.4 大模型LLM的架构介绍?

### 2.2 大模型 进阶面

  2.2.1 llama 输入句子长度理论上可以无限长吗?

  2.2.2 什么是 LLMs 复读机问题?

  2.2.3 为什么会出现 LLMs 复读机问题?

  2.2.4 如何缓解 LLMs 复读机问题?

  2.2.5 什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选?

  2.2.6 各个专业领域是否需要各自的大模型来服务?

  2.2.7 如何让大模型处理更长的文本?

### 2.3 大模型 微调面

2.3.1 如果想要在某个模型基础上做全参数微调，究竟需要多少显存?

2.3.2 为什么SFT之后感觉LLM傻了?

  2.3.3 SFT 指令微调数据如何构建?

  2.3.4 领域模型Continue PreTrain 数据选取?

  2.3.5 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力?

  2.3.6 领域模型Continue PreTrain ，如何让模型在预训练过程中就学习到更多的知识?

  2.3.7 进行SFT操作的时候，基座模型选用Chat还是Base?

  2.3.8 领域模型微调 指令&数据输入格式 要求?

  2.3.9 领域模型微调 领域评测集 构建?

  2.3.10 领域模型词表扩增是不是有必要的?

  2.3.11 如何训练自己的大模型?

  2.3.12 训练中文大模型有啥经验?

  2.3.13 指令微调的好处?

  2.3.14 预训练和微调哪个阶段注入知识的?

  2.3.15 想让模型学习某个领域或行业的知识，是应该预训练还是应该微调?

  2.3.16 多轮对话任务如何微调模型?

  2.3.17 微调后的模型出现能力劣化，灾难性遗忘是怎么回事?

  2.3.18 微调模型需要多大显存?

  2.3.19 大模型LLM进行SFT操作的时候在学习什么?

  2.3.20 预训练和SFT操作有什么不同?

  2.3.21 样本量规模增大，训练出现OOM错误?

  2.3.22 大模型LLM进行SFT 如何对样本进行优化?

  2.3.23 模型参数迭代实验?

### 2.5 大模型 langchain面

  - **概念部分**

  2.5.1 什么是LangChain

  2.5.2 LangChain 包含哪些核心概念?

  2.5.3 什么是LangChain Agent?

  2.5.4 如何使用LangChain?

  2.5.5 LangChain 支持哪些功能?

  2.5.6 什么是LangChain model?

  2.5.7 LangChain 包含哪些特点?

  2.5.8 LangChain 如何使用?

  2.5.9 LangChain 存在哪些问题及方法方案?

  2.5.10 LangChain 替代方案?

  - **基础技术部分**

  2.5.11 LangChain 中 Components and Chains 是什么?

  2.5.12 LangChain 中 Prompt Templates and Values 是什么?

  2.5.13 LangChain 中 Example Selectors 是什么?

  2.5.14 LangChain 中 Output Parsers 是什么?

  2.5.15 LangChain 中 Indexes and Retrievers 是什么?

  2.5.16 LangChain 中 Chat Message History 是什么?

  2.5.17 LangChain 中 Agents and Toolkits 是什么?

  2.5.18 LangChain 如何调用LLMs生成回复?

  2.5.19 LangChain 如何修改提示模板?

  2.5.20 LangChain 如何链接多个组件处理一个特定的下游任务?

  2.5.21 LangChain 如何Embedding & vector store?

  2.5.22 LangChain 低效的令牌使用问题

  2.5.23 LangChain 文档的问题

  2.5.24 LangChain 太多概念容易混淆，过多的“辅助”函数问题

  2.5.25 LangChain 行为不一致并且隐藏细节问题

  2.5.26 LangChain 缺乏标准的可互操作数据类型问题

### 2.6 基于LLM+向量库的文档对话 面

  2.6.1 LLMs 存在模型幻觉问题，请问如何处理?

  2.6.2 基于LLM+向量库的文档对话思路是怎么样?

  2.6.3 基于LLM+向量库的文档对话核心技术是什么?

  2.6.4 基于LLM+向量库的文档对话 prompt 模板如何构建?

### 2.7 大模型 参数高效微调(PEFT) 面

  - **LoRA篇**

  2.7.1 什么是 LoRA?

  2.7.2 LoRA 的思路是什么?

  2.7.3 LoRA 的特点是什么?

  - **QLoRA篇**

  2.7.4 QLoRA 的思路是怎么样的?

  2.7.5 QLoRA 的特点是什么?

  - **AdaLoRA篇**

  2.7.6 AdaLoRA 的思路是怎么样的?

  2.7.7 LoRA权重是否可以合入原模型?

  2.7.8 ChatGLM-6B LoRA后的权重多大?

  2.7.9 LoRA 微调优点是什么?

  2.7.10 LoRA微调方法为啥能加速训练?

  2.7.11 如何在已有LoRA模型上继续训练?

  - **提示学习（Prompting）**

  2.7.12 为什么需要 P-tuning?

  2.7.13 P-tuning 思路是什么?

  2.7.14 P-tuning 优点是什么?

  2.7.15 P-tuning 缺点是什么?

  2.7.16 为什么需要 指示微调（Prompt-tuning）?

  2.7.17 指示微调（Prompt-tuning）思路是什么?

  2.7.18 指示微调（Prompt-tuning）优点是什么?

  2.7.19 指示微调（Prompt-tuning）缺点是什么?

  2.7.20 指示微调（Prompt-tuning）与 Prefix-tuning 区别 是什么?

  2.7.21 指示微调（Prompt-tuning）与 fine-tuning 区别 是什么?

  2.7.22 提示学习（Prompting）有哪些方法，能不能稍微介绍一下它们?

  - **前缀微调（Prefix-tuning）篇**

  2.7.23 为什么需要 前缀微调（Prefix-tuning）?

  2.7.24 前缀微调（Prefix-tuning）思路是什么?

  2.7.25 前缀微调（Prefix-tuning）的优点是什么?

  2.7.26 前缀微调（Prefix-tuning）的缺点是什么?

  - **适配器微调（Adapter-tuning）篇**

  2.7.27 为什么 需要 适配器微调（Adapter-tuning）?

  2.7.28 适配器微调（Adapter-tuning）思路?

  2.7.29 适配器微调（Adapter-tuning）特点是什么?

  2.7.30 AdapterFusion 思路 是什么?

  2.7.31 AdapterDrop 思路 是什么?

  2.7.32 AdapterDrop 特点 是什么?

  2.7.33 MAM Adapter 思路 是什么?

  2.7.34 MAM Adapter 特点 是什么?

### 2.8 大模型 推理面

  2.8.1 为什么大模型推理时显存涨的那么多还一直占着?

  2.8.2 大模型在gpu和cpu上推理速度如何?

  2.8.3 推理速度上，int8和fp16比起来怎么样?

  2.8.4 大模型有推理能力吗?

  2.8.5 大模型生成时的参数怎么设置?

  2.8.6 有哪些省内存的大语言模型训练/微调/推理方法?

  2.8.7 如何让大模型输出合规化

  2.8.8 应用模式变更

### 2.9 大模型 评测面

  2.9.1 大模型怎么评测?

  2.9.2 大模型的honest原则是如何实现的?

  2.9.3 模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力?

### 2.10 大模型 强化学习面

  2.10.1 奖励模型需要和基础模型一致吗?

  2.10.2 RLHF 在实践过程中存在哪些不足?

  2.10.3 如何解决 人工产生的偏好数据集成本较高，很难量产问题?

  2.10.4 如何解决三个阶段的训练（SFT->RM->PPO）过程较长，更新迭代较慢问题?

  2.10.5 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高问题?

### 2.11 大模型 软硬件配置面

  2.11.1 介绍一下 FFN 块 计算公式?

  2.11.2 介绍一下 GeLU 计算公式?

  2.11.3 介绍一下 Swish 计算公式?

  2.11.4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式?

  2.11.5 介绍一下 使用 GeLU 的 GLU 块 计算公式?

  2.11.6 介绍一下 使用 Swish 的 GLU 块 计算公式?

  2.11.7 各LLMs 都使用哪种激活函数?

### 2.12 大模型 训练集面

  2.12.1 SFT（有监督微调）的数据集格式?

  2.12.2 RM（奖励模型）的数据格式?

  2.12.3 PPO（强化学习）的数据格式?

  2.12.4 找数据集哪里找?

  2.12.5 微调需要多少条数据?

  2.12.6 有哪些大模型的训练集?

  2.12.7 进行领域大模型预训练应用哪些数据集比较好?

  2.12.8 如何给LLM注入领域知识?

  2.12.9 如果想要快速体验各种模型，该怎么办?

### 2.13 Token及模型参数准备篇

  2.13.1 预训练数据 Token 重复 是否影响 模型性能?

  2.13.2 SFT需要训练Token数?

### 2.14 ALiBi (Attention with Linear Biases)篇

  2.14.1 ALiBi (Attention with Linear Biases) 思路是什么?

  2.14.2 ALiBi (Attention with Linear Biases) 的偏置矩阵是什么?有什么作用?

  2.14.3 ALiBi (Attention with Linear Biases) 有什么优点?

  2.14.4 ALiBi (Attention with Linear Biases) 被哪些 LLMs 应用?

### 2.15 LLMs 位置编码篇

  2.15.1 什么是位置编码?

  2.15.1.1 什么是绝对位置编码?

  2.15.1.2 什么是相对位置编码?

  - **旋转位置编码 RoPE篇**

  2.15.2 旋转位置编码 RoPE 思路是什么?

  2.15.3 推导一下 旋转位置编码 RoPE ?

  2.15.4 旋转位置编码 RoPE 有什么优点?

  2.15.5 旋转位置编码 RoPE 被哪些 LLMs 应用?

### 2.16 长度外推问题篇

  2.16.1 什么是 长度外推问题?

  2.16.2 长度外推问题 的 解决方法 有哪些?

### 2.17 LLMs Tokenizer 篇

  2.17.1 Byte-Pair Encoding(BPE)篇

  2.17.1.1 Byte-Pair Encoding(BPE) 如何构建词典?

  2.17.2 WordPiece 篇

  2.17.2.1 WordPiece 与 BPE 异同点是什么?

  2.17.3 SentencePiece 篇

  2.17.3.1 简单介绍一下 SentencePiece 思路?

  - **对比篇**

  2.17.4 举例介绍一下 不同 大模型LLMs 的分词方式?

  2.17.5 介绍一下 不同 大模型LLMs 的分词方式 的区别?

### 2.18 Layer Normalization 篇

  - **Layer Norm 篇**

  2.18.1 Layer Norm 的计算公式写一下?

  - **RMS Norm 篇 （均方根 Norm）**

  2.18.2 RMS Norm 的计算公式写一下?

  2.18.3 RMS Norm 相比于 Layer Norm 有什么特点?

  - **Deep Norm 篇**

  2.18.4 Deep Norm 有什么优点?

  2.18.5 Deep Norm 思路?

  2.18.6 写一下 Deep Norm 代码实现?

  - **Layer normalization-方法篇**

  - **Layer normalization-位置篇**

  2.18.7 LN 在 LLMs 中的不同位置 有什么区别么?如果有，能介绍一下区别么?

  - **Layer normalization 对比篇**

  2.18.8 LLMs 各模型分别用了 哪种 Layer normalization?

[![GitHub stars](https://img.shields.io/github.com/Joining-AI/LLM_Interview_Prepare?style=social)](https://github.com/Joining-AI/LLM_Interview_Prepare)