[**BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion**](https://www.alphaxiv.org/abs/2508.08241) 2025.8

* 让双足人形机器人在真实世界中稳定复现人类多样化的高动态复杂动作
* BeyondMimic：双阶段框架
  * 模仿学习训练出能精确跟踪多种人类动作的鲁棒运动策略
  * 将专家策略蒸馏进一个统一的引导式扩散模型，使其测试时zero-shot适应新任务
 
[**HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit**](https://www.alphaxiv.org/abs/2502.13013) 2025.4

* 统一的全身遥操作座舱：脚踏板，外骨骼手臂，数据手套
* 训练时逐渐增加上肢姿态的变化范围和难度
* 高度追踪奖励：激励机器人精确蹲到操作员指定的目标高度
* 利用机器人身体对称性来增强数据，翻转左右关节信息、动作指令，创造镜像的数据点
* 3D打印的同构外骨骼手臂的关节与机器人手臂的关节结构相匹配，无需逆向运动学
* 基于霍尔传感器的手套成本低，硬件成本特别低（500美元）

[**π0.5: a Vision-Language-Action Model with Open-World Generalization**](https://www.alphaxiv.org/abs/2504.16054) 2025.4

* 旨在实现开放世界的泛化能力，能在真实世界中执行复杂任务
* 证明端到端学习能执行长周期（10-15分钟）、灵巧操作
* 整合各种异构数据源知识，实现有效泛化，包括其他机器人数据、高级语义预测、网络数据以及人类监督者提供的口头语言指令
