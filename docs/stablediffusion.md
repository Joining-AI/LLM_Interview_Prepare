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

### 1.1 写在前面

<details>
  <summary class="bold-summary">为什么要为Diffusion单独开辟一个部分</summary>
首先，将Diffusion部分纳入“大模型面经”这一篇章的主要考量，首先是其同属于“AIGC”这一领域，同时，是其模型参数量和训练需求（硬件需求如GPU，服务器等，数据需求如良好标注的图片，音频，或是视频数据集）都已经达到了所谓“大模型”的门槛，一如之前介绍的那样。

这些需求都已经达到了高校难以企及的高度（both of them）。目前所有国内高校实验室，均没有能力复现StableDiffusion或是Sora，可灵这样的爆款，亦或是Open-Sora这样的开源项目。国内高校实验室的算力和数据储备，充其量完成一个基于特定模型的模块添加工作（如AnimateDiff），或是对于其某个能力的分析，对于一个较小模型的训练（绝大多数的论文）。究其原因，是由于Diffusion相关的工程结果可以快速商业化变现，变现速度甚至强于LLM。因而无论企业界还是学术界，哪怕是政界都很在关注这方面的研究，当下图像和音频扩散的理论研究已经研究的大差不差了。悲观地来讲，当下期待从实验室科研中获得真正上手实操的，SOTA级别的全流程图像或视频扩散模型的训练是完全不切实际的。

如果想要真正的上手实操，可以考虑自主复现AudioLDM，即Audio Latent Diffusion Model的代码并自行训练（建议使用luchen云之类的云服务器，或是可以参加像钱彦旻老师这样的课题组来获得相关经历）。这是由于音频相对显存要求较低，并且相关研究已经开展很久，数据集易采集且详尽。

其次想说的点是，请不要将“使用过Stable Diffusion 的ComfyUI”或是对Diffusion原理有一些粗浅理解就宣称掌握了Diffusion，哪怕是真的大量使用过SD，真的“很有prompt使用心得”，真的“很了解其数学概念”，或是上了哪边的“AI使用课程”。这些“自信”会给前沿开发者或是科研人员带来强烈不适。当然，有相关使用经验是一件锦上添花的事情，但也请务必扎根具体原理和代码给出更务实且严谨，有深度的理解。
</details>

接下来是Diffusion相关原理部分。

### 1.2 各类 Diffusion 模型原理篇

请注意，Diffusion模型是一个数学上很好理解的模型，这里只提供部分的常见内容及其较浅的讲解。深度理解建议查看原论文。

1.2.1 DDPM算法原理部分：
<details>
  <summary><strong>简述DDPM的算法流程：</strong></summary>

1. 初始化：一般指去噪（模型推理）过程的初始化，指从带噪声的图像的设定开始（或纯高斯噪声）。

2. 正向扩散：逐步向数据添加高斯噪声，直到数据完全转化为无结构的噪声。
$$x_{t+1}=\alpha_{t} x_{t}+\beta_{t} \epsilon_{t}, \ \epsilon_{t} \sim \mathcal{N}(0,\mathcal{I})$$
$$x_{t}=\bar{\alpha_t} x_{0}+\bar{\beta_t} \epsilon, \ \epsilon \sim \mathcal{N}(0,\mathcal{I})$$
（关键点理解：(2)式可由(1)式逐步推得；其中，由式子可以看出，下一个时间步 $t$ 时刻的图像$x_{t}$是关于上一步图像$x_{t-1}$的高斯分布，$\alpha_{t}，\beta_{t}$为预设超参，满足$\alpha_{t}+\beta_{t}=1$，同时$\alpha_{t}$递减，$\bar{\alpha_t}$为$\alpha_{0}$到$\alpha_{t}$的累乘结果，一般来讲，存在线性递减或余弦递减两种超参方案，前者为DDPM初始论文的设置，后者为原作者在Improved DDPM论文中的对比设置）

3. 反向去噪：通过模型预测并逐渐去掉每一步加入的噪声，还原得到无噪声的图像。
$$p_{\theta}(x_{t-1}|x_{t})=\mathcal{N}(x_{t-1},\mu_{\theta}(x_{t},t),\Sigma_{\theta}(x_{t},t))$$
（关键点理解：此处，$\theta$代表神经网络，下表意味为“神经网络预测结果”，这里通过神经网络预测上一时刻$x_{t-1}$的高斯分布，预测其均值和方差。而实际上，根据扩散过程的定义，我们可以推导均值为：$$\mu_{\theta}(x_{t},t)=\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{\beta_{t}}{1-\bar{\alpha_{t}}}\Sigma_{\theta}(x_{t},t))$$，因而这里神经网络本质上是在预测噪声，也是许多人直接成预训练扩散模型神经网络为“预训练去噪器”的一个原因；具体推理过程中，直接将之前对应时刻的图像替换成推理得到的分布中任意抽取结果当作上一时刻真实图像）

4. 训练：使用反向传播算法更新模型参数，以最小化正向和反向过程之间的差异，即最小化每一时刻原始正向扩散的分布和反向扩散预测的分布之间的差异，也即每一时刻添加的实际噪声和预测的噪声结果的差异。

5. 测试：对新的高噪声图像应用训练好的模型进行去噪。
</details>

<details>
  <summary><strong>实现DDPM需要什么条件？</strong></summary>
	1. 马尔可夫链：DDPM使用马尔可夫链来描述数据的扩散过程。马尔可夫链是一个随机过程，具有无记忆性，即在给定当前状态的情况下，未来的状态只依赖于当前状态。
	   这一点可以由上一点中的公式看出，上或下一时刻的图像结果只取决于当前时刻的图像和预设的超参。
	2. 微小变化：DDPM通过逐步添加微小的高斯噪声来扩散数据。这些微小的变化是在数据中引入随机性的关键步骤。这由预设的超参实现。具体来讲，在公式中体现为$\beta_{t}$，即方差规模的大小
	3. 高斯噪声变化：DDPM使用高斯噪声来模拟数据的扩散过程。高斯噪声是一种常见的随机噪声，也称为正态分布噪声。
</details>

<details>
  <summary><strong>为什么DDPM加噪声的幅度是不一致的？</strong></summary>
	前期加噪少是为了保持数据结构的完整性，后期加噪多是为了加速扩散过程，使得模型能够更快地从噪声中恢复出清晰的数据。
	这一点在原论文作者Improved DDPM论文中有具体阐述
</details>

<details>
  <summary><strong>DDPM预测噪声还是预测当前分布？</strong></summary>
	由之前推导，本质上预测噪声，预测分布只是中间过程
</details>

1.2.2 DDIM算法原理部分：
<details>
  <summary><strong>DDIM是怎么实现加速采样的？</strong></summary>
1. 确定性逆扩散

在DDPM中，逆扩散过程是随机的。这意味着，在从某个噪声状态 $x_t$ 计算前一状态 $x_{t-1}$ 时，会引入随机噪声。具体来说，这一步涉及从一个条件高斯分布中采样。
DDIM对这一过程做出了修改，去除了随机性，使逆扩散过程完全确定性。在DDIM中，$x_{t-1}$ 的计算直接依赖于 $x_t$ 和一个从网络学习到的噪声预测 $\epsilon_\theta(x_t, t)$。这种改动使得生成过程在给定相同的起始噪声和模型参数的情况下总是产生相同的结果，提高了过程的可重复性。

2. 非马尔科夫跳步采样

DDPM的采样过程是严格遵循时间顺序的，即每个时间步的输出依赖于其直接前一个时间步的状态，形成一个马尔科夫链。而DDIM允许在时间步之间进行更大的跳跃。这意味着可以直接从 $x_t$ 跳到 $x_{t-s}$（其中 $s > 1$），跳过一些中间状态。这种非马尔科夫的跳步采样大大减少了需要计算的总步骤数，从而加快了整个采样过程。

3. 加速生成过程

由于DDIM中的逆扩散是确定性的，并且支持跳步采样，它能显著加快生成过程。在实际应用中，如实时图像生成和视频处理等场景，这种加速非常有价值。

4. 牺牲了一定的多样性

虽然DDIM在效率和确定性方面有所增加，但这是以牺牲一定的输出多样性为代价的。在DDPM中，随机采样帮助探索多种可能的生成路径，从而增加了生成数据的多样性。DDIM的确定性路径意味着对于同一起始噪声，输出将总是相同的，这限制了模型输出的多样性。

5. 理论上的调整

在理论上，DDIM还通过对扩散和逆扩散方程的重新参数化（例如，通过改变噪声水平的方程）使其更适合非马尔科夫跳步采样和确定性逆扩散。这些改动涉及数学上的深入调整，以确保模型即使在大步长跳跃的情况下也能保持高质量的输出。
</details>

<details>
  <summary><strong>DDIM是不是确定性生成，为什么？</strong></summary>
	是确定性⽣成。因为在逆向去噪声过程中，DDIM的逆推公式，将随机噪声的部分置为0。具体来讲，原先生成过程相当于在得到每一步结果后基于该时间步的分布会进行一次随机采样，改进后取消了该随机采样，因而对于预训练完成的模型来讲，对于给定时间$t$和图像$x_{t}$，生成的指定时间步图像固定
</details>
1.2.3 Score-Based-diffusion-model：

<details>
  <summary><strong>提供了什么解释扩散模型的等价方式？</strong></summary>
- 提供了⼀种解释扩散模型的等价⽅式，其中降噪过程可以看作是沿着分数（梯度）前进。
- 这也是本整理者比较推崇的一种扩散模型解释。可以理解为，预训练扩散模型是一张导引地图，指示旅行者寻找路径攀上高山（概率高峰）的导引图
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
优点：泛化性能强，特征在同⼀空间下衡量，模型简单不需要额外训练。

缺陷：⽂本描述简单“A photo of a xxx”，图⽂理解能⼒偏弱
</details>

<details>
  <summary><strong>介绍一下BLIP/BLIP2的原理？</strong></summary>
BLIP：通过多路损失函数，以及图像分快理解策略等算法，构建⾼质量的图像理解模型。

BLIP2：在BLIP基础上，利用Q-Former构建图像与⼤语⾔模型之间的桥梁，充分利⽤⼤语⾔模型⾃身的预训练能⼒
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
	从训练集中选取一张加噪过的图片和噪声强度
	输入unet，让unet预测噪声图
	计算和真正的噪声图之间的误差
	通过反向传播更新unet的参数
	
</details>

<details>
  <summary><strong>VAE为什么会导致图像变模糊？</strong></summary>
  "VAE编解码整体是一个有损过程，可以选择减少损失，比如优化模型结构，提升采样效率等。完全不减少损失的方案就是原图反贴"
</details>

<details>
  <summary><strong>介绍一下SD，Dall-E2两者的异同？</strong></summary>
Dalle2通过自回归的方式逐个预测像素点，最终生成符合描述的图像。

SD加⼊了Latent-Space（⼤幅降低特征维度），以及交叉注意⼒机制+Unet的步骤，更精细更可控
</details>

<details>
  <summary><strong>介绍一下classifier-free guidance和Classifier Guidance？</strong></summary>
Classifier Guidance的一般流程如下：首先单独预训练一个噪声鲁棒的分类器模型。然后训练一个普通的无条件Diffusion模型。Diffusion模型生成图像的采样过程中,利 用预训练好的分类器来提供条件信号。具体来说,就是每个采样步骤都计算分类器的输 出,获得条件影响项,加入到Diffusion模型的更新公式中。这样就可以利用分类器的条 件信号,实现Diffusion模型在推理阶段条件生成图像的目的。
	
Classifier-Free Guidance 中，⽣成模型不仅仅学习如何根据给定的条件⽣成数据，⽽且还学习如何在没有任何条件输⼊的情况下⽣成数据。换句话说，模型既能进⾏条件⽣成，也能进⾏⽆条件⽣成。CFG的训练过程其实就是对提供的条件输入做随机的dropout，这样就可以得到一个无条件和条件提示的两个输出，然后学习二者之间的方向差指导采样过程。在⽣成过程中，Classifier-Free Guidance 允许我们在没有显式使⽤分类器或判别器的情况下调节条件⽣成的强度。这是通过“调节”（或“混合”）条件⽣成和⽆条件⽣成的输出来实现的，以此来控制⽣成内容的相关性和多样性 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)    
guidance scale是一个放缩系数，越大，生成的结果越倾向于输入条件，多样性会下降。 越小，多样性越大。
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
	VAE：将图像特征/⽂本特征，映射到Latent Space。
	LDM相关：Diffusion Model +Unet，去噪声核⼼步骤
	Conditioning：作⽤于Unet的 Cross-Attention位置，实现对输出结果的控制

</details>

<details>
  <summary><strong>为什么原生SD的控制效果不太好，需要引入如ControlNet的控制模型？</strong></summary>
  "因为控制是一个隐性控制模型，通过CrossAttention的权重隐性引导生成结果，并不是完全控制"
</details>

1.2.7 SDXL篇：
<details>
  <summary><strong>SDXL的核心优化？</strong></summary>
	接⼊级联的refiner模型+微调⽹络结构，⼤幅度提升⽣成质量。
	多样化的训练策略，⼤幅提升基础模型表达能⼒
</details>

<details>
  <summary><strong>SDXL的训练策略？</strong></summary>

图像尺⼨条件化：把图像的尺⼨编码后作为信息输⼊到模型中。
裁剪参数化训练：裁剪坐标也和尺⼨⼀样送⼊模型中。
多尺度训练：多尺度+分桶
噪声偏置：针对冷⻔⾊域，加⼊初始化噪声偏置
</details>

1.2.8 Diffusion模型微调篇：

- Lora：

	核⼼解读关键词：低秩展开，即插即⽤
	通过矩阵低秩展开，使⽤“外接”低秩展开后的⽹络对原模型进⾏更新

- Lora有没有什么优化方案？

	Locon/loha，分别进⾏细节质量和速度存储空间的优化

- DreamBooth：

	核⼼解读关键词：正则化微调整个⽹络，训练数据混合
	因为使⽤正则化，只在预训练⽹络上微调某类特定的case。 所以速度反⽽⽐Lora快得多

1.2.9 Textual Inversion（知识点）：

- 关键词：⽂本embedding，Transformer
- 核⼼总结：通过对Embedding层的特殊编码，实现通过不同输⼊⽂本，来影响模型最终的⽣成结果。影响的是Embedding的部分
- 首先需要定义一个在现有模型中没有的关键词，新的关键词会和其他的关键词一样，生成Tokenizer(用不同的数字表示)；然后将其转换为embedding；text transformer会映射出对于新给的关键词最好的embedding向量。不用改变模型，可以看作在模型中寻找新的表征来表示新的关键字


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
	复制原⽣Unet⼀样的模型结构，前半部分encoder训练，后半部分⽤Zero Convolution 承接，decoder部分接⼊到模型Unet的⽹络层中
	“Zero Convolution”即零卷积：是带有零初始化权重和偏差的1×1卷积。在进⾏⾃⼰的模型训练开始之前，所有零卷积输出都是零，此时模型仍然是原始的Stable Diffusion Model
</details>

<details>
  <summary><strong>T2I Adapter?</strong></summary>
  每张条件图片都会别额外编码，编码信息会被加入到 UNET 噪声预测中训练时候，冻结了原先的 unet，只对 Adapter 部分进行单独训练。
</details>

<details>
  <summary><strong>IP-Adapter?</strong></summary>
  IP-Adapter 通过带有解耦交叉注意力的适配模块，将文本特征的 Cross-Attention 和图像特征的 Cross-Attention 区分开来，在 Unet 的模块中新增了一路 Cross-Attention 模块，用于引入图像特征。
</details>


[![GitHub stars](https://img.shields.io/github/stars/Joining-AI/LLM_Interview_Prepare?style=social)](https://github.com/Joining-AI/LLM_Interview_Prepare)