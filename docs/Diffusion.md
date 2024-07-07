<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Markdown with CSS</title>
    <link rel="stylesheet" type="text/css" href="src/styles.css">
</head>
<body>

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
