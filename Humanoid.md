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


