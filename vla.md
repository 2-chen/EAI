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

[**UniVLA: Learning to Act Anywhere with Task-centric Latent Actions**](https://www.alphaxiv.org/abs/2505.06111) 2025.5

[**Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success**](https://www.alphaxiv.org/abs/2502.19645) 2025.4

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

[AUTORT: EMBODIED FOUNDATION MODELS FOR LARGE SCALE ORCHESTRATION OF ROBOTIC AGENTS](https://www.alphaxiv.org/abs/2401.12963)

* 大规模、自动化地在真实世界中收集机器人操作数据的框架
* 利用视觉模型观察环境，利用语言模型自主提出有意义、可执行的任务
* 机器人宪法：LLM生成和筛选任务时要遵循的规则，为部署机器人提供安全保障
* AutoRT：
   * 探索：自主寻找可能有操作价值的场景
   * 任务生成：使用VLM进行场景描述，场景描述和机器人宪法一起输入LLM中，生成包含多个潜在操作任务的列表
   * 可供性过滤：根据可行性和安全性筛选任务
   * 数据收集：选择任务后执行，若需要人类帮助会向远程操作员发出请求，记录操作任务的过程

