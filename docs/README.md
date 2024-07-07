<style>
  .label {
      background-color: #80EB00;
      color: black;
      padding: 3px;
      border-radius: 12px; /* 使方框为圆角 */
  }
  details summary {
    list-style-type: disc;  /* 添加圆点序号 */
    margin-left: 20px;      /* 增加与圆点的间距 */
    font-weight: normal;    /* 确保文本不加粗 */
    cursor: pointer;        /* 鼠标悬浮时显示为可点击状态 */
  }

  details summary strong {
    font-weight: normal;    /* 确保 strong 标签中的文本不加粗 */
  }

  /* 鼠标悬浮时显示淡蓝色底色 */
  details summary:hover {
    background-color: rgb(192, 229, 241);
  }

  /* 用于加粗的类 */
  .bold-summary {
    font-weight: bold;
  }
</style>

# LLM_Interview_Prepare

maintained by [SJTU Joining AI](https://sjtujoining.com) attribute to Malotru

本仓库秉承开源精神，关于大模型面试中常见面试试题和面试经验的整理，同时维护大模型相关的工作与求职机会。

# 零 写在前面

## [AI面试与职业机会](Part0/Part0-0.html)

## [markdown文档格式模板](Part0/Part0-1.html)

# 第一部分 基础知识

## [什么是大模型](Part1/Part1-1.html)

## [大模型类别](Part1/Part1-2.html)


# 第二部分：大模型一般知识点
（主要内容借鉴于多个网站的面经资料）

## LLM面经


### 2.1 大模型（LLMs）基础面

  2.1.1 目前主流的开源模型体系有哪些?

  2.1.2 prefix LM 和 causal LM 区别是什么?

  2.1.3 涌现能力是啥原因?

  2.1.4 大模型LLM的架构介绍?

### 2.2 大模型（LLMs）进阶面

  2.2.1 llama 输入句子长度理论上可以无限长吗?

  2.2.2 什么是 LLMs 复读机问题?

  2.2.3 为什么会出现 LLMs 复读机问题?

  2.2.4 如何缓解 LLMs 复读机问题?

  2.2.5 什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选?

  2.2.6 各个专业领域是否需要各自的大模型来服务?

  2.2.7 如何让大模型处理更长的文本?

### 2.3 大模型（LLMs）微调面

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

### 2.5 大模型（LLMs）langchain面

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

### 2.7 大模型（LLMs）参数高效微调(PEFT) 面

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

### 2.8 大模型（LLMs）推理面

  2.8.1 为什么大模型推理时显存涨的那么多还一直占着?

  2.8.2 大模型在gpu和cpu上推理速度如何?

  2.8.3 推理速度上，int8和fp16比起来怎么样?

  2.8.4 大模型有推理能力吗?

  2.8.5 大模型生成时的参数怎么设置?

  2.8.6 有哪些省内存的大语言模型训练/微调/推理方法?

  2.8.7 如何让大模型输出合规化

  2.8.8 应用模式变更

### 2.9 大模型（LLMs）评测面

  2.9.1 大模型怎么评测?

  2.9.2 大模型的honest原则是如何实现的?

  2.9.3 模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力?

### 2.10 大模型（LLMs）强化学习面

  2.10.1 奖励模型需要和基础模型一致吗?

  2.10.2 RLHF 在实践过程中存在哪些不足?

  2.10.3 如何解决 人工产生的偏好数据集成本较高，很难量产问题?

  2.10.4 如何解决三个阶段的训练（SFT->RM->PPO）过程较长，更新迭代较慢问题?

  2.10.5 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高问题?

### 2.11 大模型（LLMs）软硬件配置面

  2.11.1 介绍一下 FFN 块 计算公式?

  2.11.2 介绍一下 GeLU 计算公式?

  2.11.3 介绍一下 Swish 计算公式?

  2.11.4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式?

  2.11.5 介绍一下 使用 GeLU 的 GLU 块 计算公式?

  2.11.6 介绍一下 使用 Swish 的 GLU 块 计算公式?

  2.11.7 各LLMs 都使用哪种激活函数?

### 2.12 大模型（LLMs）训练集面

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




## Diffusion 面经

### 1.1 写在前面

<details>
  <summary class="bold-summary">为什么要为Diffusion单独开辟一个部分</summary>
首先，将Diffusion部分纳入“大模型面经”这一篇章的主要考量，首先是其同属于“AIGC”这一领域，同时，是其模型参数量和训练需求（硬件需求如GPU，服务器等，数据需求如良好标注的图片，音频，或是视频数据集）都已经达到了所谓“大模型”的门槛，一如之前介绍的那样。

这些需求都已经达到了高校难以企及的高度（both of them）。目前所有国内高校实验室，均没有能力复现StableDiffusion或是Sora，可灵这样的爆款，亦或是Open-Sora这样的开源项目。国内高校实验室的算力和数据储备，充其量完成一个基于特定模型的模块添加工作（如AnimateDiff），或是对于其某个能力的分析，对于一个较小模型的训练（绝大多数的论文）。究其原因，是由于Diffusion相关的工程结果可以快速商业化变现，变现速度甚至强于LLM。因而无论企业界还是学术界，哪怕是政界都很在关注这方面的研究，当下图像和音频扩散的理论研究已经研究的大差不差了。悲观地来讲，当下期待从实验室科研中获得真正上手实操的，SOTA级别的全流程图像或视频扩散模型的训练是完全不切实际的。

如果想要真正的上手实操，可以考虑自主复现AudioLDM，即Audio Latent Diffusion Model的代码并自行训练（建议使用luchen云之类的云服务器，或是可以参加像钱彦旻老师这样的课题组来获得相关经历）。这是由于音频相对显存要求较低，并且相关研究已经开展很久，数据集易采集且详尽。

其次想说的点是，请不要将“使用过Stable Diffusion 的ComfyUI”或是对Diffusion原理有一些粗浅理解就宣称掌握了Diffusion，哪怕是真的大量使用过SD，真的“很有prompt使用心得”，真的“很了解其数学概念”，或是上了哪边的“AI使用课程”。这些“自信”会给前沿开发者或是科研人员带来强烈不适。当然，有相关使用经验是一件锦上添花的事情，但也请务必扎根具体原理和代码给出更务实且严谨，有深度的理解。
</details>

接下来是Diffusion相关原理部分。

### 1.2 各类Diffusion模型原理篇
1.2.1 DDPM算法原理部分：
<details>
  <summary><strong>简述DDPM的算法流程：</strong></summary>
  "初始化：从带噪声的图像开始。正向扩散：逐步向数据添加高斯噪声，直到数据完全转化为无结构的噪声。反向去噪：通过模型预测并逐渐去掉每一步加入的噪声，还原得到无噪声的图像。训练：使用反向传播算法更新模型参数，以最小化正向和反向过程之间的差异。测试：对新的高噪声图像应用训练好的模型进行去噪。"
</details>

<details>
  <summary><strong>实现DDPM需要什么条件？</strong></summary>
  "马尔可夫链：DDPM使用马尔可夫链来描述数据的扩散过程。马尔可夫链是一个随机过程，具有无记忆性，即在给定当前状态的情况下，未来的状态只依赖于当前状态。微小变化：DDPM通过逐步添加微小的高斯噪声来扩散数据。这些微小的变化是在数据中引入随机性的关键步骤。高斯噪声变化：DDPM使用高斯噪声来模拟数据的扩散过程。高斯噪声是一种常见的随机噪声，也称为正态分布噪声。"
</details>

<details>
  <summary><strong>为什么DDPM加噪声的幅度是不一致的？</strong></summary>
  "前期加噪少是为了保持数据结构的完整性，后期加噪多是为了加速扩散过程，使得模型能够更快地从噪声中恢复出清晰的数据。"
</details>

<details>
  <summary><strong>DDPM预测噪声还是预测当前分布？</strong></summary>
  "预测噪声，预测分布只是中间过程"
</details>

1.2.2 DDIM算法原理部分：
<details>
  <summary><strong>DDIM是怎么实现加速采样的？</strong></summary>
  "DDIM通过保证DDPM的三项前向条件不变：前向高斯噪声+马尔可夫链，实现逆向递推公式优化，减少逆向推理步骤"
</details>

<details>
  <summary><strong>DDIM是不是确定性生成，为什么？</strong></summary>
  "是确定性生成。因为在逆向去噪声过程中，DDIM的逆推公式，将随机噪声的部分置为0"
</details>
1.2.3 Score-Based-diffusion-model：

<details>
  <summary><strong>提供了什么解释扩散模型的等价方式？</strong></summary>
  "提供了一种解释扩散模型的等价方式，其中降噪过程可以看作是沿着分数（梯度）前进"
</details>
1.2.4 高阶采样方案：
<details>
  <summary><strong>是否了解DPM++等加速采样方案？</strong></summary>
  "通过ODE对扩散模型进行建模，通过解析解的形式解构扩散模型求解步骤"
</details>

1.2.5 特征编码篇：

<details>
  <summary><strong>介绍一下CLIP编码？</strong></summary>
  "构建大规模的图像-文本数据构建（文本，图像）pair对，在其他下游子任务中取得极高的zero-shot指标"
</details>

<details>
  <summary><strong>CLIP编码特征的优缺点？</strong></summary>
  "优点：泛化性能强，特征在同一空间下衡量，模型简单不需要额外训练。缺陷：文本描述简单'A photo of a xxx'，图文理解能力偏弱"
</details>

<details>
  <summary><strong>介绍一下BLIP/BLIP2的原理？</strong></summary>
  "BLIP：通过多路损失函数，以及图像分块理解策略等算法，构建高质量的图像理解模型。BLIP2：在BLIP基础上，利用Q-Former构建图像与大语言模型之间的桥梁，充分利用大语言模型自身的预训练能力"
</details>

<details>
  <summary><strong>为什么BLIP/BLIP2的特征没法直接用？</strong></summary>
  "因为受到图文一致性等隐形损失约束，相关特征不再同一个特征空间下（无法直接用距离衡量图文特征的相似性）。因此无法像CLIP一样“直接”接入模型中使用"
</details>

1.2.6 Stable Diffusion篇：

<details>
  <summary><strong>Stable Diffusion的核心优化是什么？</strong></summary>
  "通过VAE将特征映射到Latent Space，大幅减少运算量的同时还能保证生成质量。通过Unet实现对生成内容的引导"
</details>

<details>
  <summary><strong>Stable Diffusion是怎么训练的？</strong></summary>
  "从训练集中选取一张加噪过的图片和噪声强度输入unet，让unet预测噪声图计算和真正的噪声图之间的误差通过反向传播更新unet的参数"
</details>

<details>
  <summary><strong>VAE为什么会导致图像变模糊？</strong></summary>
  "VAE编解码整体是一个有损过程，可以选择减少损失，比如优化模型结构，提升采样效率等。完全不减少损失的方案就是原图反贴"
</details>

<details>
  <summary><strong>介绍一下SD，Dall-E2两者的异同？</strong></summary>
  "Dalle2通过自回归的方式逐个预测像素点，最终生成符合描述的图像。SD加入了Latent-Space（大幅降低特征维度），以及交叉注意力机制+Unet的步骤，更精细更可控"
</details>

<details>
  <summary><strong>介绍一下classifier-free guidance和Classifier Guidance？</strong></summary>
  "Classifier Guidance的一般流程如下：首先单独预训练一个噪声鲁棒的分类器模型。然后训练一个普通的无条件Diffusion模型。Diffusion模型生成图像的采样过程中,利用预训练好的分类器来提供条件信号。具体来说,就是每个采样步骤都计算分类器的输出,获得条件影响项,加入到Diffusion模型的更新公式中。这样就可以利用分类器的条件信号,实现Diffusion模型在推理阶段条件生成图像的目的。Classifier-Free Guidance 中，生成模型不仅仅学习如何根据给定的条件生成数据，而且还学习如何在没有任何条件输入的情况下生成数据。换句话说，模型既能进行条件生成，也能进行无条件生成。CFG的训练过程其实就是对提供的条件输入做随机的dropout，这样就可以得到一个无条件和条件提示的两个输出，然后学习二者之间的方向差指导采样过程。在生成过程中，Classifier-Free Guidance 允许我们在没有显式使用分类器或判别器的情况下调节条件生成的强度。这是通过“调节”（或“混合”）条件生成和无条件生成的输出来实现的，以此来控制生成内容的相关性和多样性 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) guidance scale是一个放缩系数，越大，生成的结果越倾向于输入条件，多样性会下降。 越小，多样性越大。"
</details>

<details>
  <summary><strong>Stable Diffusion怎么实现文本和图像的条件控制？</strong></summary>
  "文本/图像编码器将文本/图像信息编码，然后通过交叉注意力机制将信息引入扩散模型。SD 的 U-Net 既用到了自注意力，也用到了交叉注意力。自注意力用于图像特征自己内部信息聚合。交叉注意力用于让生成图像对齐文本，其 Q 来自图像特征，K, V 来自文本编码"
</details>
<details>
  <summary><strong>扩散模型添加时间步timestep信息？</strong></summary>
  "通过类似于Transformer中的位置编码方法，将常数转换为向量并添加到输入图像中"
</details>

<details>
  <summary><strong>Noise Scheduler了解吗？</strong></summary>
  "Noise Scheduler定义了一个高斯分布，其均值和方差随着时间步的变化而变化，以控制噪声的添加量"
</details>

<details>
  <summary><strong>Stable Diffusion核心模块有哪些？</strong></summary>
  "VAE：将图像特征/文本特征，映射到Latent Space。LDM相关：Diffusion Model +Unet，去噪声核心步骤Conditioning：作用于Unet的 Cross-Attention位置，实现对输出结果的控制"
</details>

<details>
  <summary><strong>为什么原生SD的控制效果不太好，需要引入如ControlNet的控制模型？</strong></summary>
  "因为控制是一个隐性控制模型，通过CrossAttention的权重隐性引导生成结果，并不是完全控制"
</details>

1.2.7 SDXL篇：
<details>
  <summary><strong>SDXL的核心优化？</strong></summary>
  "接入级联的refiner模型+微调网络结构，大幅度提升生成质量。多样化的训练策略，大幅提升基础模型表达能力"
</details>

<details>
  <summary><strong>SDXL的训练策略？</strong></summary>
  "图像尺寸条件化：把图像的尺寸编码后作为信息输入到模型中。裁剪参数化训练：裁剪坐标也和尺寸一样送入模型中。多尺度训练：多尺度+分桶噪声偏置：针对冷门色域，加入初始化噪声偏置"
</details>

1.2.8 Diffusion模型微调篇：
- Lora：

  "核心解读关键词：低秩展开，即插即用通过矩阵低秩展开，使用“外接”低秩展开后的网络对原模型进行更新"

- Lora有没有什么优化方案？

  "Locon/loha，分别进行细节质量和速度存储空间的优化"

- DreamBooth：

  "核心解读关键词：正则化微调整个网络，训练数据混合因为使用正则化，只在预训练网络上微调某类特定的case。所以速度反而比Lora快得多"

1.2.9 Textual Inversion（知识点）：

- 关键词：文本embedding，Transformer
- 核心总结：
  "通过对Embedding层的特殊编码，实现通过不同输入文本，来影响模型最终的生成结果。影响的是Embedding的部分"

1.2.10 Lora/Dreambooth/Textual Inversion，核心差异点：
- Lora：
  "是小模型即插即用微调。"
- Dreambooth：
  "大模型特化全量微调"
- Textual Inversion：
  "Text-embedding 编码修改"

1.2.11 控制模型篇：



<details>
  <summary><strong>介绍一下ControlNet的核心原理？</strong></summary>
  "复制原生Unet一样的模型结构，前半部分encoder训练，后半部分用Zero Convolution 承接，decoder部分接入到模型Unet的网络层中'Zero Convolution'即零卷积：是带有零初始化权重和偏差的1×1卷积。在进行自己的模型训练开始之前，所有零卷积输出都是零，此时模型仍然是原始的Stable Diffusion Model"
</details>

<details>
  <summary><strong>T2I Adapter?</strong></summary>
  每张条件图片都会别额外编码，编码信息会被加入到 UNET 噪声预测中训练时候，冻结了原先的 unet，只对 Adapter 部分进行单独训练。
</details>

<details>
  <summary><strong>IP-Adapter?</strong></summary>
  IP-Adapter 通过带有解耦交叉注意力的适配模块，将文本特征的 Cross-Attention 和图像特征的 Cross-Attention 区分开来，在 Unet 的模块中新增了一路 Cross-Attention 模块，用于引入图像特征。
</details>
