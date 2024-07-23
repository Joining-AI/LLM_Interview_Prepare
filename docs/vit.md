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

  /* 展开后显示更淡的蓝色底色 */
  details[open] summary {
    background-color: rgb(240, 248, 255);
  }

  /* 展开的内容显示更淡的蓝色底色 */
  details[open] {
    background-color: rgb(240, 248, 255);
    padding: 10px; /* 增加内边距以提高可读性 */
    border-radius: 5px; /* 使边角稍微圆润 */
  }
</style>


[![GitHub stars](https://img.shields.io/github/stars/Joining-AI/LLM_Interview_Prepare?style=social)](https://github.com/Joining-AI/LLM_Interview_Prepare)

事实上，目前很少有能称为“大模型”的多模态模型。但依照现在(2024.7)的发展规模和速度，很快就会涌现出一大批参数量上逼近大模型门槛的模型，这也是添加其到“大模型面试准备”的一个重要原因。

这里整理一系列与“多模态大模型”主题相关的面经资料和相关预备知识

### $基本预备知识$：

<details>
  <summary><strong>CLIP模型</strong></summary> 
（1）基本思想：文本和图像在特征域进行对齐

（2）模型结构：
- 文本编码器（BERT）
-  图像编码器（ViT）
-  对上面两个encoder提取的特征计算余弦距离对齐

（3）训练目标：info-nce-loss (对比学习MoCo的对抗基因)

（4）训练数据集：400million的图文对齐图片数据
</details>

<details>
  <summary><strong>BLIP模型</strong></summary>
（1）基本思想：深度嵌入联合编码器，同时使三个视觉语言模型上联合表现最佳：图像对文本对比学习、图像文本互翻译联合条件生成训练。

（2）模型结构：Multimodal mixture of Encoder-Decoder
- 多模态交叉编码器（ViT）
- 串联文本编码器（BERT）
- 串联Image-grounded text encoder: 在self-attention和FFN中间加一层cross-attention
- 多模态Image-grounded text decoder: 用casual self-attention层（预测下一个token）代替了双向自注意力层（建立当前输入token的表达）

（3）训练目标：
- ITC loss（视图距离，生成图像和文本编码器距离）；
- ITM loss（视图距离，生成图像和文本编码器）； 
- LM loss（语言生成，优化图像和文本编码器在语言建模应用可互式交互生成的效果。
这三种损失（loss）是在多模态（图文）模型中常用的，用以优化图文对齐和生成性能。以下是对这三种损失的详细介绍：

1. **ITC loss (Image-Text Contrastive loss)**
   - **目的**：优化图文对齐的编码器，改善图像和文本之间的语义匹配。
   - **工作原理**：这种损失函数通常通过对比学习实现，即通过最大化匹配图文对的相似性，同时最小化不匹配对的相似性。实现方式通常涉及计算图像和文本嵌入之间的距离或相似度，并使用如负对数似然这样的方法进行优化。
   - **应用**：这种方法在提升图文嵌入在同一向量空间中的表现上非常有效，有助于后续的检索和分类任务。

2. **ITM loss (Image-Text Matching loss)**
   - **目的**：优化图文匹配能力，用于更细致地判定图像内容与文本描述之间的匹配程度。
   - **工作原理**：这种损失通常用于二分类任务，即判断给定的图像和文本是否匹配。模型需要学习区分哪些图文组合是真实相关的，哪些是随机组合的。这通常通过一个有监督的学习过程来完成，其中正例是实际匹配的图文对，而负例是不匹配的图文对。
   - **应用**：这有助于模型在更复杂的多模态场景下进行准确的内容理解和生成，常用于自动图文生成、图文同步解读等应用。

3. **LM loss (Language Modeling loss)**
   - **目的**：主要用于语言生成任务，尤其是在图文交互生成中的应用，通过优化模型的语言生成能力来增强交互性。
   - **工作原理**：这种损失函数通常用于评估模型生成的文本与实际文本之间的一致性。在图文模型中，可以使用图像作为上下文来生成描述文本，并通过计算生成文本和真实描述之间的交叉熵损失来优化模型。
   - **应用**：这种方法使得模型不仅能够理解图像内容，还能基于理解生成准确、自然的语言描述，常用于图像描述、自动文案生成等任务。

（4）训练数据集：
- COCO
- Visual Genome
- 网络数据：Conceptual Captions 3M,  Conceptual 12M（视图数据），SBU Captions
- 另外进行一个额外的拍摄照片文本对齐的web模型数据 LAION (115M 图片)
</details>

<details>
  <summary><strong>BLIP2模型</strong></summary>
（1）基本思想：两个阶段，通过利用预训练好的视觉模型和语言模型来提升多模态效果和降低训练成本。

（2）模型结构：
	BLIP-2 由预训练的Image Encoder，预训练的Large Language Model，和一个可学习的 Q-Former 组成。
-  Image Encoder：从输入图片中提取视觉特征，尝试了两种网络结构，CLIP 训练的 ViT-L/14和EVA-CLIP训练的 ViT-g/14（去掉了最后一层）。
- Large Language Model：大语言模型进行文本生成，尝试了接入decoder-based LLM 和 encoder-decoder-based LLM两种结构。
- Q-Former：弥补视觉和语言两种模态的modality gap，可以理解为固定图像编码器和固定LLM之间的**信息枢纽**，选取最有用的视觉特征给LLM来生成文本。

（3） 训练目标：
- ITC loss（偏理解）：图文对比学习，对其图文特征空间；
- ITG loss（偏生成）：确定输入图像生成文本描述，迫使Query提取包含文本信息的特征；
- ITM loss（偏理解）：图文匹配二分类，图文表示的细粒度对齐。

（4）训练数据
- BLIP运用的数据集
- 利用提出的CapFilt方法从网络图中提取匹配的图文对，然后对文字描述内容进行filter，得到有利于训练的caption
</details>

4. BEIT方法


### $常见面经题目$

1. CLIP和BEIT V3的区别;
2. BEIT V3除了BERT外还有别的特殊的设计吗;
3. V3和V2的 embbeding有什么不同;
4. VIT的patch怎么做的;
5. 224* 224 * 3的图像做成14 * 14的patch的话，最后sequence的长度是多少;
6. Transformer里的position encoding怎么做的;
7. 相对位置编码和绝对位置编码有什么区别吗;
8. 具体实现相对位置编码该怎么做;
9. CV领域有哪些其他的预训练的模型;
10. 什么是对比学习;
11. coco,simclr等对比学习方法是怎么具体做的；

[![GitHub stars](https://img.shields.io/github/stars/Joining-AI/LLM_Interview_Prepare?style=social)](https://github.com/Joining-AI/LLM_Interview_Prepare)