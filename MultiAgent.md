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

