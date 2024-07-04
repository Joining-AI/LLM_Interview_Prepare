# LLM_Interview_Prepare

maintained by [SJTU Joining AI](https://sjtujoining.com) attribute to Malotru

本仓库是关于大模型面试中常见面试试题和面试经验的整理，同时维护大模型相关的工作与求职机会。

### [AI面试与职业机会](Part0/Part0-0.html)


# 第一部分 基础知识

## [什么是大模型](Part1/Part1-1.html)

## [大模型类别](Part1/Part1-2.html)


# 第二部分：大模型一般知识点
（主要内容借鉴于多个网站的面经资料）

## LLM面经

-  **大模型（LLMs）基础面**
    
     1. 目前 主流的开源模型体系 有哪些？
     2. prefix LM 和 causal LM 区别是什么？
     3. 涌现能力是啥原因？
     4. 大模型LLM的架构介绍？   
     
- **大模型（LLMs）进阶面**
    
     1. llama 输入句子长度理论上可以无限长吗？
        
     1. 什么是 LLMs 复读机问题？
        
     2. 为什么会出现 LLMs 复读机问题？
        
     3. 如何缓解 LLMs 复读机问题？
        
     4. 什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？
        
     5. 各个专业领域是否需要各自的大模型来服务？
        
     6. 如何让大模型处理更长的文本？
        
-  **大模型（LLMs）微调面**
    
     1. 如果想要在某个模型基础上做全参数微调，究竟需要多少显存？
        
     2. 为什么SFT之后感觉LLM傻了?
        
     3. SFT 指令微调数据 如何构建?
       
     4. 领域模型Continue PreTrain 数据选取？
        
     5. 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力？
        
     6. 领域模型Continue PreTrain ，如何 让模型在预训练过程中就学习到更多的知识？
        
     7. 进行SFT操作的时候，基座模型选用Chat还是Base?
        
     8. 领域模型微调 指令&数据输入格式 要求？
        
     9. 领域模型微调 领域评测集 构建？
        
     10. 领域模型词表扩增是不是有必要的？
        
     11. 如何训练自己的大模型？
        
	12. 训练中文大模型有啥经验？
        
     13. 指令微调的好处？
        
     14. 预训练和微调哪个阶段注入知识的？
        
     15. 想让模型学习某个领域或行业的知识，是应该预训练还是应该微调？
        
     16. 多轮对话任务如何微调模型？
        
     17. 微调后的模型出现能力劣化，灾难性遗忘是怎么回事？
        
     18. 微调模型需要多大显存？
        
     19. 大模型LLM进行SFT操作的时候在学习什么？
        
     20. 预训练和SFT操作有什么不同
        
     21. 样本量规模增大，训练出现OOM错
        
     22. 大模型LLM进行SFT 如何对样本进行优化？
        
     23. 模型参数迭代实验
    
        
-  **大模型（LLMs）langchain面**
    
    1. 基于LLM+向量库的文档对话 基础面
     
     - **概念部分**
     
	     1. 什么是 LangChain?
        
	     2. LangChain 包含哪些 核心概念？
        
	     3. 什么是 LangChain Agent?
        
	     4. 如何使用 LangChain ?
        
	     5. LangChain 支持哪些功能?
        
	     6. 什么是 LangChain model?
        
	     7. LangChain 包含哪些特点?
        
	     8. LangChain 如何使用?
        
	     9. LangChain 存在哪些问题及方法方案？
        
	     10. LangChain 替代方案？

	- **基础技术部分**
	
          1. LangChain 中 Components and Chains 是什么？
          
          2. LangChain 中 Prompt Templates and Values 是什么？
          
          3. LangChain 中 Example Selectors 是什么？
          
          4. LangChain 中 Output Parsers 是什么？
          
          5. LangChain 中 Indexes and Retrievers 是什么？
          
          6. LangChain 中 Chat Message History 是什么？
          
          7. LangChain 中 Agents and Toolkits 是什么？

          8. LangChain 如何调用 LLMs 生成回复？
          
          9. LangChain 如何修改 提示模板？
          
          10. LangChain 如何链接多个组件处理一个特定的下游任务？
          
          11. LangChain 如何Embedding & vector store？
          
          12. LangChain 低效的令牌使用问题
          
          13. LangChain 文档的问题
          
          14. LangChain 太多概念容易混淆，过多的“辅助”函数问题
          
          15. LangChain 行为不一致并且隐藏细节问题
          
          16. LangChain 缺乏标准的可互操作数据类型问题
    
              
    2. **基于LLM+向量库的文档对话 优化面**
    
	     1. LLMs 存在模型幻觉问题，请问如何处理？
        
	     2. 基于LLM+向量库的文档对话 思路是怎么样？
        
	     3. 基于LLM+向量库的文档对话 核心技术是什么？
        
	     4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
        
	     - 1. 痛点1：文档切分粒度不好把控，既担心噪声太多又担心语义信息丢失
        
	     - 2. 痛点2：在基于垂直领域 表现不佳
        
	     - 3. 痛点3：langchain 内置 问答分句效果不佳问题
        
	     - 4. 痛点4：如何 尽可能召回与query相关的Document 问题
        
	     - 5. 痛点5：如何让LLM基于query和context得到高质量的respose
        
	        
-  **大模型（LLMs）参数高效微调(PEFT) 面**
    
     1. LoRA篇
     
          1.1 什么是 LoRA？
        
          1.2 LoRA 的思路是什么？
        
          1.3 LoRA 的特点是什么？
        
     2. QLoRA篇
            
          2.1 QLoRA 的思路是怎么样的？
        
          2.2 QLoRA 的特点是什么？
     
     3. AdaLoRA篇
     
          3.1 AdaLoRA 的思路是怎么样的？
        
     4. LoRA权重是否可以合入原模型？
        
     5. ChatGLM-6B LoRA后的权重多大？
        
     6. LoRA 微调优点是什么？
        
     7. LoRA微调方法为啥能加速训练？
        
     8. 如何在已有LoRA模型上继续训练？
        
     9. 什么是 提示学习（Prompting）？
           
          9.1  为什么需要 P-tuning？
               
          9.2 P-tuning 思路是什么？
              
          9.3 P-tuning 优点是什么？
        
          9.4 P-tuning 缺点是什么？
        
          9.5 为什么需要 指示微调（Prompt-tuning）？
        
          9.6 指示微调（Prompt-tuning）思路是什么？
        
          9.7 指示微调（Prompt-tuning）优点是什么？
        
          9.8 指示微调（Prompt-tuning）缺点是什么？
        
          9.9 指示微调（Prompt-tuning）与 Prefix-tuning 区别 是什么？
        
          9.10 指示微调（Prompt-tuning）与 fine-tuning 区别 是什么？
        
     11. 提示学习（Prompting）有哪些方法，能不能稍微介绍一下它们？
     
     12. 前缀微调（Prefix-tuning）篇
     
     - 11.1 为什么需要 前缀微调（Prefix-tuning）？
     - 11.2 前缀微调（Prefix-tuning）思路是什么？
     - 11.3 前缀微调（Prefix-tuning）的优点是什么？
     - 11.4 前缀微调（Prefix-tuning）的缺点是什么？
        
        
     12. 适配器微调（Adapter-tuning）篇
        
          12.1 为什么 需要 适配器微调（Adapter-tuning）？
        
          12.2 适配器微调（Adapter-tuning）思路？
        
          12.3 适配器微调（Adapter-tuning）特点是什么？
        
          12.4 AdapterFusion 思路 是什么？
        
          12.5 AdapterDrop 思路 是什么？
        
          12.6 AdapterDrop 特点 是什么？
        
          12.7 MAM Adapter 思路 是什么？
        
          12.8 MAM Adapter 特点 是什么？
        
**大模型（LLMs）推理面**
    
     1. 为什么大模型推理时显存涨的那么多还一直占着？
       
     2. 大模型在gpu和cpu上推理速度如何？
        
     3. 推理速度上，int8和fp16比起来怎么样？
        
     4. 大模型有推理能力吗？
        
     5. 大模型生成时的参数怎么设置？
        
     6. 有哪些省内存的大语言模型训练/微调/推理方法？
        
     7. 如何让大模型输出合规化
        
     8. 应用模式变更
        
 **大模型（LLMs）评测面**
    
     1. 大模型怎么评测？
        
     2. 大模型的honest原则是如何实现的？
        
     3. 模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力？
        
 **大模型（LLMs）强化学习面**
    
     1. 奖励模型需要和基础模型一致吗？
        
     2. RLHF 在实践过程中存在哪些不足？
        
     3. 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
      
     4. 如何解决三个阶段的训练（SFT->RM->PPO）过程较长，更新迭代较慢问题？
        
     5. 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？
        
**大模型（LLMs）软硬件配置面**
	  
	1. 介绍一下 FFN 块 计算公式？
        
     2. 介绍一下 GeLU 计算公式？
        
     3. 介绍一下 Swish 计算公式？
        
     4. 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？
        
     5. 介绍一下 使用 GeLU 的 GLU 块 计算公式？
        
     6. 介绍一下 使用 Swish 的 GLU 块 计算公式？
        
     7. 各LLMs 都使用哪种激活函数？
    
 **大模型（LLMs）训练集面**
    
     1. SFT（有监督微调）的数据集格式？
        
     2. RM（奖励模型）的数据格式？
        
     3. PPO（强化学习）的数据格式？
        
     4. 找数据集哪里找？
        
     5. 微调需要多少条数据？
        
     6. 有哪些大模型的训练集？
        
     7. 进行领域大模型预训练应用哪些数据集比较好？
    
     8.  如何给LLM注入领域知识？
        
     9. 如果想要快速体验各种模型，该怎么办？
        
 **Token及模型参数准备篇**
    
     1. 预训练数据 Token 重复 是否影响 模型性能？
        
     2. SFT需要训练Token数？
        
 **ALiBi (Attention with Linear Biases)篇**
    
     1. ALiBi (Attention with Linear Biases) 思路是什么？
        
     2. ALiBi (Attention with Linear Biases) 的偏置矩阵是什么？有什么作用？
       
     3. ALiBi (Attention with Linear Biases) 有什么优点？
        
     4. ALiBi (Attention with Linear Biases) 被哪些 LLMs 应用？

 **LLMs 位置编码篇**
        
     1. 什么是位置编码？
 
        1.1 什么是绝对位置编码？
        1.2  什么是相对位置编码？
        
     2.  旋转位置编码 RoPE篇
 
	    2.1  旋转位置编码 RoPE 思路是什么？
        
	    2.2 推导一下 旋转位置编码 RoPE ？
        
	    2.3 旋转位置编码 RoPE 有什么优点？
        
	    2.4 旋转位置编码 RoPE 被哪些 LLMs 应用？
        
        
 **长度外推问题篇**
 
     1. 什么是 长度外推问题？
        
     2.  长度外推问题 的 解决方法 有哪些？
        
 **LLMs Tokenizer 篇**
    
     1. Byte-Pair Encoding(BPE)篇
     Byte-Pair Encoding(BPE) 如何构建词典？
        
     2. WordPiece 篇
      WordPiece 与 BPE 异同点是什么？
        
     3. SentencePiece 篇
        简单介绍一下 SentencePiece 思路？
        
     4. 对比篇
        4.1 举例 介绍一下 不同 大模型LLMs 的分词方式？
        4.2 介绍一下 不同 大模型LLMs 的分词方式 的区别？
        
**LLMs Tokenizer 篇**
        
- **Layer Normalization 篇**
      
     - Layer Norm 篇
     
     Layer Norm 的计算公式写一下？
        
     - RMS Norm 篇 （均方根 Norm）

         RMS Norm 的计算公式写一下？
        
         RMS Norm 相比于 Layer Norm 有什么特点？
        
    - Deep Norm 篇
        
	     Deep Norm 有什么优点？
        
	     Deep Norm 思路？
        
	     写一下 Deep Norm 代码实现？
        
	-  Layer normalization-方法篇
        
	     - Layer normalization-位置篇
	     
	       LN 在 LLMs 中的不同位置 有什么区别么？如果有，能介绍一下区别么？
     
	     - Layer normalization 对比篇
	      
	       LLMs 各模型分别用了 哪种 Layer normalization？


## Diffusion 面经

### 一、写在前面
	
首先，将Diffusion部分纳入“大模型面经”这一篇章的主要考量，首先是其同属于“AIGC”这一领域，同时，是其模型参数量和训练需求（硬件需求如GPU，服务器等，数据需求如良好标注的图片，音频，或是视频数据集）都已经达到了所谓“大模型”的门槛，一如之前介绍的那样。

这些需求都已经达到了高校难以企及的高度（both of them）。目前所有国内高校实验室，均没有能力复现StableDiffusion或是Sora，可灵这样的爆款，亦或是Open-Sora这样的开源项目。国内高校实验室的算力和数据储备，充其量完成一个基于特定模型的模块添加工作（如AnimateDiff），或是对于其某个能力的分析，对于一个较小模型的训练（绝大多数的论文）。究其原因，是由于Diffusion相关的工程结果可以快速商业化变现，变现速度甚至强于LLM。因而无论企业界还是学术界，哪怕是政界都很在关注这方面的研究，当下图像和音频扩散的理论研究已经研究的大差不差了。悲观地来讲，当下期待从实验室科研中获得真正上手实操的，SOTA级别的全流程图像或视频扩散模型的训练是完全不切实际的。

如果想要真正的上手实操，可以考虑自主复现AudioLDM，即Audio Latent Diffusion Model的代码并自行训练（建议使用luchen云之类的云服务器，或是可以参加像钱彦旻老师这样的课题组来获得相关经历）。这是由于音频相对显存要求较低，并且相关研究已经开展很久，数据集易采集且详尽。

其次想说的点是，请不要将“使用过Stable Diffusion 的ComfyUI”或是对Diffusion原理有一些粗浅理解就宣称掌握了Diffusion，哪怕是真的大量使用过SD，真的“很有prompt使用心得”，真的“很了解其数学概念”，或是上了哪边的“AI使用课程”。这些“自信”会给前沿开发者或是科研人员带来强烈不适。当然，有相关使用经验是一件锦上添花的事情，但也请务必扎根具体原理和代码给出更务实且严谨，有深度的理解。

接下来是Diffusion相关原理部分。

二、各类Diffusion模型原理篇（以下为基础常识类问题）

1. DDPM算法原理部分：

- 简述DDPM的算法流程：

	初始化：从带噪声的图像开始。
	正向扩散：逐步向数据添加高斯噪声，直到数据完全转化为无结构的噪声。
	反向去噪：通过模型预测并逐渐去掉每一步加入的噪声，还原得到无噪声的图像。
	训练：使用反向传播算法更新模型参数，以最小化正向和反向过程之间的差异。
	测试：对新的高噪声图像应用训练好的模型进行去噪。
	
- 实现DDPM是否需要什么条件：

	马尔可夫链：DDPM使用马尔可夫链来描述数据的扩散过程。马尔可夫链是一个随机过程，具有无记忆性，即在给定当前状态的情况下，未来的状态只依赖于当前状态。
	微小变化：DDPM通过逐步添加微小的高斯噪声来扩散数据。这些微小的变化是在数据中引入随机性的关键步骤。
	高斯噪声变化：DDPM使用高斯噪声来模拟数据的扩散过程。高斯噪声是一种常见的随机噪声，也称为正态分布噪声。
	
- 为什么DDPM加噪声的幅度是不⼀致的？

	前期加噪少是为了保持数据结构的完整性，后期加噪多是为了加速扩散过程，使得模型能够更快地从噪声中恢复出清晰的数据。

- DDPM预测噪声还是预测当前分布？

	预测噪声，预测分布只是中间过程

2. DDIM算法原理部分： 

- DDIM是怎么实现加速采样的？

	DDIM通过保证DDPM的三项前向条件不变：前向⾼斯噪声+⻢尔可夫链，实现逆向递推公式优化，减少逆向推理步骤

- DDIM是不是确定性⽣成，为什么 

	是确定性⽣成。因为在逆向去噪声过程中，DDIM的逆推公式，将随机噪声的部分置为0


3. Score-Based-diffusion-model

- 提供了⼀种解释扩散模型的等价⽅式，其中降噪过程可以看作是沿着分数（梯度）前进

 4. ⾼阶采样⽅案：
 
- 是否了解DPM++等加速采样⽅案
	
	通过ODE对扩散模型进⾏建模，通过解析解的形式解构扩散模型求解步骤

5. 特征编码篇：

- 介绍⼀下CLIP编码：

	构建⼤规模的图像-⽂本数据构建（⽂本，图像）pair对，在其他下游⼦任务中取得极⾼的zero-shot指标

- CLIP编码特征的优缺点

	优点：泛化性能强，特征在同⼀空间下衡量，模型简单不需要额外训练。

	缺陷：⽂本描述简单“A photo of a xxx”，图⽂理解能⼒偏弱

- 介绍⼀下BLIP/BLIP2的原理

	BLIP：通过多路损失函数，以及图像分快理解策略等算法，构建⾼质量的图像理解模型。

	BLIP2：在BLIP基础上，利用Q-Former构建图像与⼤语⾔模型之间的桥梁，充分利⽤⼤语⾔模型⾃身的预训练能⼒

- 为什么BLIP/BLIP2的特征没法直接⽤

	因为受到⽂图⼀致性等隐形损失约束，相关特征不再同⼀个特征空间下（⽆法直接⽤距离衡量⽂图特征的相似性）。因此⽆法像CLIP⼀样“直接”接⼊模型中使⽤ 

6. Stable Diffusion篇： 

- Stable Diffusion 的核⼼优化是什么？

	通过VAE将特征映射到Latent Space，⼤幅减少运算量的同时还能保证⽣成质量。
	通过Unet实现对⽣成内容的引导

- Stable Diffusion是怎么训练的？

	从训练集中选取一张加噪过的图片和噪声强度
	输入unet，让unet预测噪声图
	计算和真正的噪声图之间的误差
	通过反向传播更新unet的参数
	
- VAE为什么会导致图像变模糊 

	VAE编解码整体是⼀个有损过程，可以选择减少损失，⽐如优化模型结构，提升采样效率等。完全不减少损失的⽅案就是原图反贴 

- 介绍⼀下SD，Dall-E2两者的异同

	Dalle2通过自回归的方式逐个预测像素点，最终生成符合描述的图像。

	SD加⼊了Latent-Space（⼤幅降低特征维度），以及交叉注意⼒机制+Unet的步骤，更精细更可控

- 介绍下classifier-free guidance和Classifier Guidance

	Classifier Guidance的一般流程如下：首先单独预训练一个噪声鲁棒的分类器模型。然后训练一个普通的无条件Diffusion模型。Diffusion模型生成图像的采样过程中,利 用预训练好的分类器来提供条件信号。具体来说,就是每个采样步骤都计算分类器的输 出,获得条件影响项,加入到Diffusion模型的更新公式中。这样就可以利用分类器的条 件信号,实现Diffusion模型在推理阶段条件生成图像的目的。
	
	Classifier-Free Guidance 中，⽣成模型不仅仅学习如何根据给定的条件⽣成数据，⽽且还学习如何在没有任何条件输⼊的情况下⽣成数据。换句话说，模型既能进⾏条件⽣成，也能进⾏⽆条件⽣成。CFG的训练过程其实就是对提供的条件输入做随机的dropout，这样就可以得到一个无条件和条件提示的两个输出，然后学习二者之间的方向差指导采样过程。在⽣成过程中，Classifier-Free Guidance 允许我们在没有显式使⽤分类器或判别器的情况下调节条件⽣成的强度。这是通过“调节”（或“混合”）条件⽣成和⽆条件⽣成的输出来实现的，以此来控制⽣成内容的相关性和多样性 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)    
	guidance scale是一个放缩系数，越大，生成的结果越倾向于输入条件，多样性会下降。 越小，多样性越大。
	
- Stable Diffusion 怎么实现⽂本和图像的条件控制的

	⽂本/图像编码器将⽂本/图像信息编码，然后通过交叉注意⼒机制将信息引⼊扩散模型。SD 的 U-Net 既用到了自注意力，也用到了交叉注意力。自注意力用于图像特征自己内部信息聚合。交叉注意力用于让生成图像对齐文本，其 Q 来自图像特征，K, V 来自文本编码 

- 扩散模型添加时间步timestep信息

	 通过类似于Transformer中的位置编码⽅法，将常数转换为向量并添加到输⼊图像中

- Noise Scheduler了解吗 

	Noise Scheduler定义了⼀个⾼斯分布，其均值和⽅差随着时间步的变化⽽变化，以控制噪声的添加量 

- Stable Diffusion核⼼模块有哪些

	VAE：将图像特征/⽂本特征，映射到Latent Space。
	LDM相关：Diffusion Model +Unet，去噪声核⼼步骤
	Conditioning：作⽤于Unet的 Cross-Attention位置，实现对输出结果的控制

- 为什么原⽣SD的控制效果不太好，需要引⼊如ControlNet的控制模型 

	因为控制是⼀个隐性控制模型，通过CrossAttention的权重隐性引导⽣成结果，并不是完全控制 

 7. SDXL篇：

- SDXL的核⼼优化

	接⼊级联的refiner模型+微调⽹络结构，⼤幅度提升⽣成质量。
	多样化的训练策略，⼤幅提升基础模型表达能⼒

- SDXL的训练策略：

	图像尺⼨条件化：把图像的尺⼨编码后作为信息输⼊到模型中。
	裁剪参数化训练：裁剪坐标也和尺⼨⼀样送⼊模型中。
	多尺度训练：多尺度+分桶
	噪声偏置：针对冷⻔⾊域，加⼊初始化噪声偏置

8. 模型微调篇：

 - Lora：
 
	 核⼼解读关键词：低秩展开，即插即⽤
	通过矩阵低秩展开，使⽤“外接”低秩展开后的⽹络对原模型进⾏更新
	
- Lora有没有什么优化⽅案

	Locon/loha，分别进⾏细节质量和速度存储空间的优化

- DreamBooth

	核⼼解读关键词：正则化微调整个⽹络，训练数据混合
	因为使⽤正则化，只在预训练⽹络上微调某类特定的case。 所以速度反⽽⽐Lora快得多

9. Textual Inversion （知识点）

- 关键词：⽂本embedding，Transformer
- 核⼼总结：通过对Embedding层的特殊编码，实现通过不同输⼊⽂本，来影响模型最终的⽣成结果。影响的是Embedding的部分
- 首先需要定义一个在现有模型中没有的关键词，新的关键词会和其他的关键词一样，生成Tokenizer(用不同的数字表示)；然后将其转换为embedding；text transformer会映射出对于新给的关键词最好的embedding向量。不用改变模型，可以看作在模型中寻找新的表征来表示新的关键字

10. Lora/Dreambooth/Textual Inversion，核⼼差异点
- Lora：是⼩模型即插即⽤微调。
- Dreambooth：⼤模型特化全量微调
- Textual Inversion：Text-embedding 编码修改

11. 控制模型篇：

- 介绍⼀下ControlNet的核⼼原理

	复制原⽣Unet⼀样的模型结构，前半部分encoder训练，后半部分⽤Zero Convolution 承接，decoder部分接⼊到模型Unet的⽹络层中
	“Zero Convolution”即零卷积：是带有零初始化权重和偏差的1×1卷积。在进⾏⾃⼰的模型训练开始之前，所有零卷积输出都是零，此时模型仍然是原始的Stable Diffusion Model
	
 12. 适配器篇：
 
 - T2I Adapter
 
	 每张条件图片都会别额外编码，编码信息会被加入到 UNET 噪声预测中
	 训练时候，冻结了原先的 unet，只对 Adapter 部分进行单独训练
	 
- IP-Adapter

	IP-Adapter 通过带有解耦交叉注意力的适配模块，将文本特征的 Cross-Attention 和图像特征的 Cross-Attention 区分开来，在 Unet 的模块中新增了一路 Cross-Attention 模块，用于引入图像特征


 


