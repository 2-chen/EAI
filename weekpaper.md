

[**Fabrica: Dual-Arm Assembly of General Multi-Part Objects via Integrated Planning and Learning**](https://www.alphaxiv.org/abs/2506.05168) 2025.6

* 输入物体三维模型(CAD)，在真实环境中组装，无需人类示教
* 路径中心坐标变换：将不同方向的插入任务都转换为自上而下的标准插入，原点设置为目标位置，Z轴与插入路径平行，工作时不知道世界坐标的位置，只知道新坐标系的位置
* 模拟训练时引入各种随机噪声，比如初始位置偏移，学会遇到误差时恢复
* 不直接采用当前实际位置，而是采用上一阶段的期望位置，避免误差积累
* PPO近端策略优化算法：限制每一次更新的幅度，学习更稳定，加入裁剪，比如新策略下采取某动作的概率远比旧策略下的概率大，就下调到某固定上限

[**π0: A Vision-Language-Action Flow Model for General Robot Control**](https://www.alphaxiv.org/abs/2410.24164) seminar 2024.11

* 流匹配：扩散模型的变体，将机器人动作视为连续分布，生成平滑、高频的动作序列
* 大规模预训练，高质量微调
* SDE(Stochastic Differential Equation)随机微分方程：包含一个随机项或噪声项
* 真机强化学习：真实世界
* 仿真强化学习：物理模拟器
