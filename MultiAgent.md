[**COMAS: CO-EVOLVING MULTI-AGENT SYSTEMS VIA INTERACTION REWARDS**](https://www.alphaxiv.org/abs/2510.08529) 2025.10

* 完全源于智能体之间交互的内在奖励机制
* 交互：提出方案，进行批判，给出评分，评估和评分不是一个智能体，有智能体会结合方案和批判给出评分
* 奖励：零和博弈，高分表示方案正确且批判无效，低分表示方案错误且批判有价值，同时奖励高质量方案和高质量批判
* REINFORCE++：
  * KL散度惩罚：惩罚当前模型和原始预训练模型的差异，防止过拟合
  * 优势归一化：减去平均值除以标准差，保持模型更新的稳定，优势信号更平滑
  * 限制单次参数更新的幅度，防止因为一次高奖励进行破坏式更新

*想法*：完全通过多智能体内部交互实现迭代，没有引入外部数据和反馈，本质上是对内部数据进行更深层次的耦合，把方案、评估、评分对应的数据深度耦合，非常依赖本身的数据规模。Transformer是对所有数据进行耦合，这是更高层次、更有选择性的耦合，类似于特调，所以是否还可以引入更多层次的耦合，提高面对特定任务的质量？

[**Multi-Actor Multi-Critic Deep Deterministic Reinforcement Learning with a Novel Q-Ensemble Method**](https://www.alphaxiv.org/abs/2510.01083v1) 2025.10

* 同时训练多个“执行家”和多个“评论家”
* 双目标选择执行动作：预期回报和创意性
* 得分取所有评论家评分的中位数
* actor学习时每一轮让不同critic评分，以前是一个actor和一个critic配对

[**Conflict-Based Search and Prioritized Planning for Multi-Agent Path Finding Among Movable Obstacles**](https://www.alphaxiv.org/abs/2509.26050) 2025.9


[**Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control**](https://www.alphaxiv.org/abs/2509.14431) 2025.9


[**Synergy Over Spiral: A Logistics 5.0 Game-Theoretic Model for Trust-Fatigue Co-regulation in Human-Cobot Order Picking**](https://www.alphaxiv.org/abs/2508.03765v3) 2025.9

* 将“信任度”和“疲劳度”整合到一个博弈论框架中，用于优化人机协作
* 疲劳：根据人类和机器人的共同行为进行累积，高努力会增加疲劳，而机器人高协作可以帮助恢复疲劳
* 信任：机器人的行为帮助人类减轻了疲劳则增加信任；发生故障或策略不匹配则减小信任
* 信任和疲劳是相互影响的，值得信赖的机器可以降低人的疲劳，而人的疲劳也会影响对机器人的信任
* 信任协同循环：机器的有效帮助增加人的信任，高信任激励人的努力，提高效率
* 道歉机制：发生事故导致信任下降后，机器人会提供高水平协作来修复关系，提高韧性
* 动态的Stackelberg博弈：领导者先行动，并公布自己的决策，跟随者观察领导者的决策后做出反应，领导者在决策时已经预知跟随者会如何理性应对自己的决策，所以可以引导跟随者
* 机器人是领导者，观察人类的状态来决定自己的协作水平，人类是跟随者，观察机器人行动后结合自身状态来决定努力程度


[**Curriculum Imitation Learning of Distributed Multi-Robot Policies**](https://www.alphaxiv.org/abs/2509.25097) 2025.9

* 应用课程学习提升长期协调能力：在训练中逐步增加专家演示轨迹长度，帮助机器人学习和执行需要长远规划和协调的复杂任务
* 从全局数据中估计机器人的“自我中心视角”：选择邻居，坐标转换，注入噪声

[**Multi-CAP: A Multi-Robot Connectivity-Aware Hierarchical Coverage Path Planning Algorithm for Unknown Environments**](https://www.alphaxiv.org/abs/2509.14941v2) 2025.9

* 多机器人在未知环境中进行覆盖路径规划，通过分层策略高效地协调多个机器人，以最小化总路径长度和机器人间的冲突
* 高层全局规划：环境被初步划分为一个粗粒度的网格，每个节点是一个子区域，相邻节点之间有边连接
* 如果障碍物阻挡了两个子区域的连接，就移除对应的边，如果一个子区域被障碍物分割成多个不连通的部分，就会被拆分成新的节点，确保了子区域内部的连通性
* 车辆路径问题：有一个中心仓库，和许多个需要派送包裹的客户
  * 所有车辆从一个中心出发
  * 每个客户都必须被访问到，且只能被访问一次
  * 车辆完成任务后都必须返回中心仓库
  * 目标是最小化总成本（比如总行驶距离、总时间或总油耗）
* 多车场车辆路径问题：有多个仓库，车辆可以从任何一个仓库出发
  * 如何决定哪个仓库的车辆去服务哪些客户？
  * 如何为每个仓库的车辆规划最优路线？
  * 车辆完成任务后必须返回各自出发的仓库
* 开放式车辆路径问题：完成任务后不需要返回仓库
* 开放式多车场车辆路径问题：给定多个车场和一系列客户，为每个车场的车辆规划一组路线，使得每个客户都被访问一次，并且每条路线都从一个车场开始，在最后一个服务的客户处结束后不需要返回，最终目标是使所有车辆行驶的总距离最短
* 每个机器人独立地在其被分配到的子区域内执行覆盖任务，如果子区域是正在探索的（内部还有未知区域），机器人会采用简单的来回往复模式进行覆盖，如果子区域是已探索的（内部地图完全已知），机器人则会通过求解旅行商问题计算出一条覆盖内部所有单元的最短路径，探索不等于覆盖，探索只是知道地图，覆盖会进行作业
* 避免将其他机器人视为动态障碍物：从空间上对任务进行划分，为每个机器人分配了基本不重叠的工作区域，从根本上减少了机器人之间的相互干扰

[**Multi-agent embodied ai: Advances and future directions**](https://www.alphaxiv.org/abs/2505.05108) 2025.6

* 多智能体系统的复杂性：通信、协调、冲突解决和信任分配
* 多智能体强化学习的挑战：异步协作（智能体决策时间不一致），异构体协作（智能体能力、形态不同）
* 理论基础：需要为复杂的多智能体交互建立更坚实的理论基础，超越传统的马尔可夫博弈
* 新算法设计：现有算法多在理想化的模拟环境中开发，需要设计更能适应真实世界（如传感器噪声、延迟反馈）的稳健算法
* 高效学习：真实世界交互成本高，如何实现高效的样本学习，以及如何从模拟有效迁移到现实
* 通用框架：能处理多任务、多场景的通用多智能体具身AI框架
* 评估与验证：缺乏标准化、可复现且贴近现实的评估体系，尤其是在安全性和可靠性方面
* 马尔可夫博弈：环境的下一个状态取决于所有智能体同时执行的动作，每个智能体的奖励取决于自己的动作和其他所有智能体的动作

[**Conformal Data-driven Control of Stochastic Multi-Agent Systems under Collaborative Signal Temporal Logic Specifications**](https://www.alphaxiv.org/abs/2504.04615) 2025.4

