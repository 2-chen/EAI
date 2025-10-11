[**OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction**](https://www.alphaxiv.org/abs/2509.26633) 2025.9

* OmniRetarget：将人类的动作演示高效地转换为机器人可学习的运动轨迹
* 不仅转换动作，保留了机器人与物体、环境之间的空间和接触关系，如人类推箱子，转换后要保证机器人的手接触并推动了箱子
* 硬约束保证生成动作的物理可行性，比如杜绝穿模、脚部滑动等问题
* 硬约束条件：无碰撞，关节限制，足部防滑
* 数据增强：从一次人类演示中，生成大量训练数据，如基于一次捡箱子演示，生成捡起不同大小、不同初始位置的箱子的动作
* 由于数据质量高，可以用极简的强化学习方案训练策略，且策略在模拟环境训练好后，可以直接部署到真实的人形机器人上
* 交互网格：构建包含机器人关键点、交互物体表面采样点、以及环境表面采样点的四面体网格
* 最小化拉普拉斯形变：用拉普拉斯坐标描述网格中每个点相对于其邻居的局部关系，最小化机器人网格和人类网格之间拉普拉斯坐标的差异 

[**π0: A Vision-Language-Action Flow Model for General Robot Control**](https://www.alphaxiv.org/abs/2410.24164) 2024.11

* 流匹配：扩散模型的变体，将机器人动作视为连续分布，生成平滑、高频的动作序列
