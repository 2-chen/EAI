[**GMT: General Motion Tracking for Humanoid Whole-Body Control**](https://www.alphaxiv.org/abs/2506.14770) 2025.9

[**Generalizable Humanoid Manipulation with 3D Diffusion Policies**](https://www.alphaxiv.org/abs/2410.10803) 2025.9

[**VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation**](https://www.alphaxiv.org/abs/2509.20322) 2025.9

[**Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids**](https://www.alphaxiv.org/abs/2502.20396) 2025.9

[**HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**](https://www.alphaxiv.org/abs/2509.16757) 2025.9

[**OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction**](https://www.alphaxiv.org/abs/2509.26633) seminar 2025.9

* OmniRetarget：将人类的动作演示高效地转换为机器人可学习的运动轨迹
* 不仅转换动作，保留了机器人与物体、环境之间的空间和接触关系，如人类推箱子，转换后要保证机器人的手接触并推动了箱子
* 硬约束保证生成动作的物理可行性，比如杜绝穿模、脚部滑动等问题
* 硬约束条件：无碰撞，关节限制，足部防滑
* 数据增强：从一次人类演示中，生成大量训练数据，如基于一次捡箱子演示，生成捡起不同大小、不同初始位置的箱子的动作
* 由于数据质量高，可以用极简的强化学习方案训练策略，且策略在模拟环境训练好后，可以直接部署到真实的人形机器人上
* SMP(Skinned Multi-Person Linear Model)：参数化的三维人体模型
  * 形状参数：静态体型，高矮胖瘦
  * 姿态参数：动态姿势，关节旋转角
* 交互网格：构建包含机器人关键点、交互物体表面采样点、以及环境表面采样点的四面体网格
* 最小化拉普拉斯形变：用拉普拉斯坐标描述网格中每个点相对于其邻居的局部关系，最小化机器人网格和人类网格之间拉普拉斯坐标的差异
* PHC(Perpetual Humanoid Control)算法：优化整个动作序列，找到能最小化关键节点位置误差的完整关节角轨迹
* GMR算法：逐帧优化，同时匹配关键点的位置和朝向
* VideoMimic算法：保持关键点的相对关系，保持身体各部分的结构，不强行匹配世界坐标系的位置
* IMMA算法：先找到机器人关键点应该处于的理想位置，再根据理想位置求解机器人的实际姿态
  * 网格变形优化：最小化交互网格从人类到机器人的拉普拉斯形变
  * 逆运动学求解：根据理想关键点位置求解一个标准IK问题，找到匹配的关节角度
* OmniRetarget算法：逐帧求解带硬约束的优化问题，最小化交互网格的形变

[**BEHAVIOR ROBOT SUITE: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities**](https://www.alphaxiv.org/abs/2503.05652) 2025.8

[**Visual Imitation Enables Contextual Humanoid Control**](https://www.alphaxiv.org/abs/2505.03729) 2025.8

* 机器人通过观看视频学习在复杂真实环境中与物体互动，从真实到模拟再到真实的端到端学习框架VideoMimic
* 从视频中重建运动中的三维人体姿态和周围环境的三维几何结构
  * 使用现成的视觉模型从视频中提取每帧的人体三维姿态（SMPL模型）、二维关节点位置以及场景的稀疏点云，此时场景的尺度是不准的
    * 提取人体三维姿态：[VIMO](https://arxiv.org/pdf/2403.17346)
    * 提取二维关节点位置：[ViTPose](https://arxiv.org/pdf/2204.12484)，基于Vision Transformer
    * 提取场景的稀疏点云：[MegaSaM](https://arxiv.org/pdf/2412.04463)或[MonST3R](https://arxiv.org/pdf/2410.03825)，分析视频中连续帧之间的像素运动，估算出相机的移动轨迹和场景的三维结构（点云）
  * 联合优化：将人体姿态和场景点云进行联合优化，利用SMPL模型中的人体身高先验知识作为标尺，校正场景点云的真实物理尺度，对齐人和场景的运动轨迹，生成在真实物理尺度下完全对齐的4D世界（三维空间+时间）
    * 校验物理尺度：寻找全局缩放因子，使得缩放后的场景和人体模型最匹配，重投影误差将三维人体关节点投影回二维图像，看是否与二维关节点重合
    * 对齐人和场景：场景的缩放因子，人的局部位移和全局朝向，身体局部姿态
  * 将优化后的点云转换为轻量化的网格，与重力方向对齐
  * 将视频中人类动作重定向到机器人的身体结构上，生成机器人可以模仿的参考动作序列，最终产出一系列动作场景配对数据
* 单一的通用模型：不是多个针对特定任务的独立模型
* 场景感知追踪：用与训练好的模型作为起点，在动作场景数据上进行微调，策略的输入中加入周围环境的高度图，机器人学习根据地形来调整自己的动作
* 策略蒸馏：不再需要完整的参考动作作为输入，仅依赖机器人的本体感觉（关节位置、速度等）、局部地形高度图和简单方向指令，改善控制时延

[**BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion**](https://www.alphaxiv.org/abs/2508.08241) 2025.8

* 让双足人形机器人在真实世界中稳定复现人类多样化的高动态复杂动作
* 通过简单的引导信号来实时合成全新的动作以完成特定任务，即代价函数，比如距离，先扩散，再通过代价函数梯度下降
* BeyondMimic：双阶段框架
  * 模仿学习训练出多个专家策略，每个专家精通模仿一段特定的人类动作
  * 将专家策略蒸馏进一个统一的引导式扩散模型，即让专家策略在模拟环境中尽情表演，记录状态动作对，得到运动数据库，再用这个数据库来训练一个条件扩散模型
  * 将扩散模型部署到机器人或控制电脑上，机器人通过传感器获取当前状态，再输入到扩散模型，根据当前任务定义代价函数，执行引导式去噪
* 遇到分布外的奇特状态时倾向于静止不动
* 统一框架处理长序列动作：
  * 锚点追踪机制：不追踪所有身体部位的绝对坐标，而是选择一个锚点（通常是躯干），精确追踪锚点的姿态，其他部位追踪相对于锚点的姿态，允许了合理的全局漂移
  * 自适应采样：统计在哪些动作片段上失败率高，然后重点采样
  * 通用的奖励函数：追踪误差的惩罚项，不包含特定任务的奖励
  * 领域随机化：对模拟环境的物理参数加入微小的随机扰动

[**Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills**](https://www.alphaxiv.org/abs/2503.12533) 2025.5

[**AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control**](https://www.alphaxiv.org/abs/2505.03738) 2025.5

[**TWIST: Teleoperated Whole-Body Imitation System**](https://www.alphaxiv.org/abs/2505.02833) 2025.5

[**ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills**](https://www.alphaxiv.org/abs/2502.01143) 2025.4
 
[**HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit**](https://www.alphaxiv.org/abs/2502.13013) 2025.4

* 统一的全身遥操作座舱：脚踏板，外骨骼手臂，数据手套
* 训练时逐渐增加上肢姿态的变化范围和难度
* 高度追踪奖励：激励机器人精确蹲到操作员指定的目标高度
* 利用机器人身体对称性来增强数据，翻转左右关节信息、动作指令，创造镜像的数据点
* 3D打印的同构外骨骼手臂的关节与机器人手臂的关节结构相匹配，无需逆向运动学
* 基于霍尔传感器的手套成本低，硬件成本特别低（500美元）

[**Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration**](https://www.alphaxiv.org/abs/2502.14795) 2025.2
