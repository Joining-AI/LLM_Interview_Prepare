# LLM_Interview_Prepare

maintained by [SJTU Joining AI](https://sjtujoining.com)

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
        
     10. 提示学习（Prompting）有哪些方法，能不能稍微介绍一下它们？
     
     11. 前缀微调（Prefix-tuning）篇
     
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
 


