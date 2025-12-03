[**VLA2: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation**](https://www.alphaxiv.org/abs/2510.14902) [open](https://vla-2.github.io/) 2025.10

* 不强迫VLA“硬背”所有没见过的物体，而是通过外部工具（网络搜索、视觉处理）把“陌生的概念”转化为“熟悉的概念”，然后再让机器人执行
* 不仅是一个端到端的模型，而是一个包含规划、认知、记忆和执行的完整系统，将VLA模型作为执行的“手”，加上了“大脑”和“眼睛”来辅助
* 即时学习机制： 利用网络搜索实时获取未知物体的知识，将其转化为视觉特征（如颜色、形状），从而实现Zero-shot识别
* 双重“降维”打击（OOD转ID）：
  * 视觉上： 通过覆盖透明色块（Mask），掩盖物体复杂的表面纹理，让机器人只关注位置和轮廓，减少视觉过拟合。
  * 语言上： 将复杂的未知词汇（如具体的品牌名）替换为模型训练时见过的已知词汇（如“瓶子”、“碗”），让指令回到模型的舒适区
* 使用MM-GroundingDINO检测物体。如果检测不到，系统会自动去互联网搜索该物体的图片，生成描述性关键词（颜色、形状），辅助检测模型重新定位物体

[**SimpleVLA: Scaling VLA Training via Reinforcement Learnin**](https://www.alphaxiv.org/abs/2509.09674) 2025.9

* 在模拟环境中强化学习，减少对人类演示数据的依赖
* SimpleVLA-RL需演示示例启动，给模型基础能力进而有机会成功，而RL负责在原来轨迹的基础上不断探索
* RL不仅能学习已有行为，还能发现更优的解决方案，即更容易获得奖励
* 与环境闭环交互：根据视觉和状态生成动作，执行动作获得新的视觉和状态，指导任务结束获得轨迹
* 增强探索
  * 动态采样：确保每个训练批次都有成功和失败的轨迹，避免所有轨迹奖励都一样导致梯度消失，因为GRPO需要计算相对优势，有失败轨迹才能有相对优势，区分轨迹的优劣，进而判断什么是好轨迹
  * 提高裁剪范围：放宽对低概率动作的限制
  * 提高采样温度：增加动作选择的随机性

[**DISCRETE DIFFUSION VLA: BRINGING DISCRETE DIFFUSION TO ACTION DECODING IN VISION-LANGUAGE-ACTION POLICIES**](https://www.alphaxiv.org/abs/2508.20072) 2025.8

* 将视觉、语言和动作的生成过程统一到一个单一的Transformer模型中
* 将离散扩散模型应用于VLA的动作解码，将一整段未来动作序列看作一个整体，即一个token集合，通过从模糊到清晰的迭代过程来并行生成，类似于掩码到解码的过程，且不是按顺序解码
* 先易后难解码：模型先预测最有信心的动作指令，然后将不确定的指令重新掩码，在下一轮中集中精力去优化
* 二次重掩码：如果在后续迭代中模型对某个动作指令信心下降，该指令会被重新掩码并再次泛化，避免错误的累积

[**A Survey on Vision-Language-Action Models for Embodied AI**](https://arxiv.org/pdf/2405.14093) 2025.8

VLA 模型的研究分为了三个主要方向：

* 关键组件: 
    * 动力学学习：让模型理解物理规律，比如一个物体被推动后会如何运动。
    * 世界模型：让模型在“脑海”中对世界进行模拟和推演，从而进行更长远的规划。
* 底层控制策略: 模型的“肌肉记忆”和执行能力。这类模型接收具体的指令（如“拿起杯子”），并直接输出机器人手臂的精确动作序列。不同的实现架构：
    *   基于 Transformer 的模型。
    *   基于扩散模型 (Diffusion Models) 的策略。
    *   以及最新的基于大语言模型 (LLM-based) 的控制策略。
* 高层任务规划器: 当面对一个复杂、长期的任务时（如“打扫整个房间”），任务规划器负责将其分解成一系列简单的子任务，这些简单的子任务再交由底层的控制策略去执行。
* 未来方向:
    *   机器人基础模型: 研发一个能适用于多种不同机器人和任务的通用基础模型。
    *   更丰富的多模态融合: 未来还可能融合触觉、听觉等更多模态的信息。

[**HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model**](https://www.alphaxiv.org/abs/2503.10631) 2025.6

* HybridVLA结合扩散模型和自回归模型，既能利用扩散模型生成平滑、精确的连续动作，又能利用自回归模型的上下文推理能力
* 协同训练：让扩散模型的去噪过程和自回归模型的下一词元预测过程在共享的LLM主干中同时进行训练，使用混合损失函数
* 协同动作集成：执行任务时，生成两种动作预测，根据自回归模型预测的置信度融合两种预测结果
  * 模型在生成每个词元时都会输出一个概率分布，词元的概率就表示置信度，系统会计算整个自回归动作序列的平均置信度
  * 置信度高：自回归模型对场景语义理解很到位，将两种动作进行平均
  * 置信度低：场景可能比较复杂或需要高精度操作，此时完全依赖于扩散模型生成的动作，因为它在精度控制上更有优势
* 引入'\<BOD\> [扩散动作词元] \<EOD\>'将扩散过程巧妙地嵌入到LLM的标准序列预测流程中
* [视觉词元][语言词元][状态词元]\<BOD\>[扩散动作词元]\<EOD\>[自回归动作词元]
  * 处理'\<BOD\>'之后的部分时，被训练来执行扩散任务，将带噪声的扩散动作词元作为输入，预测噪声从而实现去噪
  * 处理'\<EOD\>'之后的部分时，切换回自回归任务，基于前面的信息包括已去噪的扩散动作，逐个输出离散的[自回归动作词元]
* 扩散模型生成连续空间中精确的浮点数值，自回归模型将连续的动作分割成有限的离散桶而导致精度丢失，所以扩散模型控制精度比自回归模型高

[**UniVLA: Learning to Act Anywhere with Task-centric Latent Actions**](https://www.alphaxiv.org/abs/2505.06111) 2025.5

[**Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success**](https://www.alphaxiv.org/abs/2502.19645) [open](https://openvla-oft.github.io/) 2025.4

* 针对 VLA 模型的优化微调（Optimized Fine-Tuning, OFT）
* 从“自回归解码”改为“并行解码” 
  * 旧方法：预测完 x 坐标 Token 再预测 y 坐标 Token。
  * 新方法：修改 Transformer 的注意力掩码（Mask），从因果掩码（Causal Mask）改为双向注意力（Bidirectional Attention）。模型一次性接收所有输入的视觉和语言特征，并并行输出所有的动作维度。
  * 优势：将动作生成的复杂度从 $O(D)$ 降低到 $O(1)$（D 为动作维度）
* 引入“动作分块” 
  * 利用并行解码的能力，模型在一次前向传播中不仅预测当前的动作，还预测未来 $K$ 个时间步的动作序列（例如 $K=8$ 或 $K=25$）。
  * 优势：减少了需要调用模型的频率，平滑了动作轨迹，并显著提升了长视距任务的成功率
* 


优势： 将动作生成的复杂度从 $O(D)$ 降低到 $O(1)$（D 为动作维度），极大提升了推理吞吐量

[**CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models**](https://www.alphaxiv.org/abs/2503.22020) 2025.3

[**Improving Vision-Language-Action Model with Online Reinforcement Learning**](https://www.alphaxiv.org/abs/2501.16664) 2025.1

[**OpenVLA: An Open-Source Vision-Language-Action Model**](https://www.alphaxiv.org/abs/2406.09246) 2024.9

[**RT-1: ROBOTICS TRANSFORMER FOR REAL-WORLD CONTROL AT SCALE**](https://arxiv.org/pdf/2212.06817) 2023.8

[**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**](https://www.alphaxiv.org/abs/2307.15818v1) 2023.8

* 证明在互联网上训练的视觉语言模型(VLM)可以变成控制机器人执行物理任务的策略模型
* 将机器人动作语言化：先将连续动作离散化为整数，再映射到语言模型词汇表成为动作Token
* 共同微调：将机器人动作数据和原始的网络图文数据混合在一起训练，避免模型忘记原有的网络知识
* 既能理解任务语义，也能负责具体执行动作

[**ROBOTIC TASK GENERALIZATION VIA HINDSIGHT TRAJECTORY SKETCHES**](https://www.alphaxiv.org/abs/2311.01977)

* 轨迹草图：介于语言指令和目标图像之间的表示方法
* 详细到能指导完成特定动作，粗略到能根据实际情况调整
* 通过画图、手势视频等方式来生成轨迹草图
* 事后轨迹标注：提取成功示例中末端执行器的三维轨迹，投影到机器人二维视角下，生成包含轨迹线的图像
* 通过轨迹草图连接语言指令和动作执行，也能从大量视频数据中提取动作信息来学习，避免数据浪费

[**AUTORT: EMBODIED FOUNDATION MODELS FOR LARGE SCALE ORCHESTRATION OF ROBOTIC AGENTS**](https://www.alphaxiv.org/abs/2401.12963)

* 大规模、自动化地在真实世界中收集机器人操作数据的框架
* 利用视觉模型观察环境，利用语言模型自主提出有意义、可执行的任务
* 机器人宪法：LLM生成和筛选任务时要遵循的规则，为部署机器人提供安全保障
* AutoRT：
   * 探索：自主寻找可能有操作价值的场景
   * 任务生成：使用VLM进行场景描述，场景描述和机器人宪法一起输入LLM中，生成包含多个潜在操作任务的列表
   * 可供性过滤：根据可行性和安全性筛选任务
   * 数据收集：选择任务后执行，若需要人类帮助会向远程操作员发出请求，记录操作任务的过程

