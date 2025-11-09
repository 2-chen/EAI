[**Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning**](https://www.alphaxiv.org/abs/2511.03591v1) 2025.11

[**Global-State-Free Obstacle Avoidance for Quadrotor Control in Air-Ground Cooperation**](https://www.alphaxiv.org/abs/2510.24315) 2025.10

[**Few-Shot Demonstration-Driven Task Coordination and Trajectory Execution for Multi-Robot Systems**](https://www.alphaxiv.org/abs/2510.15686) 2025.10

* 提出了一种新的少样本学习框架，旨在解决多机器人系统的协调与轨迹执行问题
* 通过时间图网络（TGN）学习任务无关的时间序列，并利用高斯过程（GP）进行连续的空间轨迹建模。使用光谱聚类提取关键交互依赖关系，应用图注意力网络（GAT）和门控递归单元（GRU）捕捉时间动态
  * 通过时间图网络（TGN）学习任务无关的时间序列？
  * 利用高斯过程（GP）进行连续的空间轨迹建模？
  * 使用光谱聚类提取关键交互依赖关系？
  * 应用图注意力网络（GAT）和门控递归单元（GRU）捕捉时间动态？

[**Decentralized Multi-Robot Relative Navigation in Unknown, Structurally Constrained Environments under Limited Communication**](https://www.alphaxiv.org/abs/2510.09188) 2025.10

* 去中心化的分层相对导航框架
* 高层的“拓扑共享与全局引导（TSGG）”模块让每个机器人在偶遇时交换轻量级的拓扑地图，逐步形成对环境全局结构的认知，实现战略级路径规划，避免陷入拓扑陷阱
  * 收到队友地图后，基于图像空间融合技术整合信息，形成一致的全局拓扑图
* 底层的“冲突解决与轨迹规划（CRTP）”模块则负责实时处理本地动态避障和与其他机器人的冲突，通过创新的采样式“逃逸点”策略，在线生成可行的临时目标，灵活化解局部死锁

[**It Takes Two: Learning Interactive Whole-Body Control Between Humanoid Robots**](https://www.alphaxiv.org/private/baf7f579-1353-48a2-a9a6-204624e69811) 2025.10

* 将真人双人互动动作数据转换成机器人能理解和执行的参考动作
  * 输入：捕捉真人互动的动作数据（三维骨骼和网络模型）
  * 接触检测：识别两人身体接触的部位和时间点
  * 对齐：以接触点为核心目标，优化两个机器人相对位置和姿态
* 扩展观察空间：机器人不仅知道自己的状态（关节角、速度），还能接收到伙伴机器人的状态信息（因为是在模拟环境中，直接共享）
* 奖励
  * 动作跟踪奖励：模仿自己的参考动作
  * 互动奖励：两个机器人关键部位相对距离和姿态是否与真人动作一致
  * 接触奖励：正确时间、正确身体部位产生接触力，并惩罚非预期的碰撞
 
[**CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks**](https://www.alphaxiv.org/abs/2509.14380) 2025.10

* LLMs和VLMs作为自动化教练，教会多机器人系统完成复杂的协作任务
* 利用LLM将复杂目标分解为子任务，用LLM生成奖励函数
* VLM闭环：
  * 评估：观察机器人执行任务的情况
  * 反馈：若失败会分析原因（奖励函数的学习曲线），生成建议
  * 优化：LLM接受建议并修改奖励函数
* 序贯训练：前一个子任务训练好的策略权重会用作下一个更难子任务的初始权重

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

* 多个智能体在可移动障碍物环境中寻找路径
* 冲突搜索CBS：双层算法，只在必要时解决冲突
  * 低层级为单个智能体规划最优路径
  * 高层级检测这些路径之间的冲突，并添加新的约束给低层，比如不能在时间t=5经过位置v
* 定义新的冲突：传统智能体之间的冲突AA，智能体与箱子AB，箱子与箱子BB
* PAMO(Pathfing Among Movable Obstacles)：在一个巨大的状态空间中搜索
* CBS-MOH(Movable Obstacles on High-level)：
  * 在低层级规划时智能体忽略箱子，只当作普通空间
  * 在高层级模拟，检测智能体的路径是否会导致AA或AB冲突，如果冲突会禁止导致该冲突的智能体的动作来解决
* CBS-MOL(Movable Obstacles on High-level)：
  * 低层级规划时每个智能体会考虑所有箱子的初始位置
  * 将一个为PAMO问题设计的规划器扩展到时空维度，命名为ST-PAMO，把时间加入到状态定义中
* PP-PAMO：基于优先级规划(PP)的算法，并使用能处理可移动障碍物的规划器
  * 为智能体设定固定优先级
  * 从最高优先级的智能体开始依次使用ST-PAMO规划器规划路径
  * 在低优先级智能体规划路径时，将规划好的高优先级智能体路径视为动态障碍物
* CBS是最优的，而PP不是，PP解的质量取决于优先级的分配

[**Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control**](https://www.alphaxiv.org/abs/2509.14431) 2025.9

* 为每个智能体创建一个专属的、以自身为中心的局部坐标系，使得每个智能体看到的局部景象都是一致的，x轴速度方向，y轴指向集群重心方向
* 将机器人集群建模成一个图，通过图神经网络Graphormer处理智能体之间的交互关系
  * Graphormer是为图结构数据定制的Transformer模型，计算每个节点和其他节点的注意力分数，衡量其他节点信息对当前节点决策的重要性
  * Graphormer在Transformer注意力分数的基础上增加中心度编码、距离编码、边特征编码

$$
\text{AttentionScore}_{ij} = \underbrace{\frac{Q_i \cdot K_j^T}{\sqrt{d_k}}}_{\text{内容相似度}} + \underbrace{b_{SP(i,j)}}_{\text{距离偏置}} + \underbrace{c_{ij}}_{\text{边特征偏置}}
$$

* 图神经网络Graphormer具有排列等变性，处理的是是集合而不是序列
  * 其计算是在节点和边上进行的，增加或减少智能体知识在图中增加或减少节点，计算方式是通用的，所以对智能体数量具有泛化性
* 将不同角色的智能体划分到不同的子图中处理，为不同角色学习特定的策略和通信模式
* 输入转换后的局部观测数据到一个基于角色的异构图神经网络，网络在各角色子图内部处理信息，捕捉同类智能体之间的交互模式，将子图节点信息汇成一个单一的代表该角色整体状态的向量，
* 智能体输出在局部坐标系下的动作，再通过旋转矩阵换到全局坐标系中

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

[**VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning**](https://faceong.github.io/VIKI-R/) 2025.6

* 让多种异构机器人在复杂的真实环境中，通过视觉感知进行有效协作
* VIKI-Bench基准测试平台：针对异构多智能体协作的层级化基准
  * L1：智能体激活，根据任务指令和场景图像，判断需要用哪些机器人
  * L2：任务规划，为被激活的机器人生成一个详细、可行的动作序列，不同类型的机器人有不同的动作指令集
  * L3：轨迹感知，根据每个智能体的视角预测自己和其他协作伙伴的运动轨迹
  * 建立在现有模拟器之上(RoboCasa、ManiSki113)，包含超1000种场景，6种不同类型的机器人和超1000种交互物体
* VIKI-R训练框架：结合监督微调和强化学习的二阶段训练框架
  * 思维链引导的预热：使用带有思考过程的标注数据对模型进行监督微调，学会解决问题的基本模式和逻辑
    * 带有思考的标注数据是针对该模拟环境的，使得大模型对齐该模拟环境
  * 分层奖励信号的强化学习：设计一个与VIKI-Bench三个层级完全对应的分层奖励函数，从是否选对机器人、规划是否可行、轨迹是否准确等多个维度对模型输出进行打分和激励，实现更精细的策略优化

[**Multi-agent embodied ai: Advances and future directions**](https://www.alphaxiv.org/abs/2505.05108) 2025.6

* 多智能体系统的复杂性：通信、协调、冲突解决和信任分配
* 多智能体强化学习的挑战：异步协作（智能体决策时间不一致）
* 新算法设计：现有算法多在理想化的模拟环境中开发，需要设计更能适应真实世界（如传感器噪声、延迟反馈）的稳健算法
* 通用框架：能处理多任务、多场景的通用多智能体具身AI框架
* 评估与验证：缺乏标准化、可复现且贴近现实的评估体系，尤其是在安全性和可靠性方面

[**Conformal Data-driven Control of Stochastic Multi-Agent Systems under Collaborative Signal Temporal Logic Specifications**](https://www.alphaxiv.org/abs/2504.04615) 2025.4

* 在不确定环境中为多智能体系统设计控制策略的新方法，智能体需要协同完成遵循复杂时序逻辑规则的任务
* 无分布假设的数据驱动：不假设随机噪声服从特定分布，利用保形预测从收集到的干扰数据中学习其影响范围
  * 保形预测：创建一个具有严格概率保证的预测区间或预测域，不需要对数据的概率分布做假设
  * 例：有95%的把握，明天的气温会在22度到28度之间
  * 计算分数并找到阈值：在校准集上，计算每个数据点的奇怪程度分数并进行排序，若想要95%置信度，就找到排在95%分位的分数，作为阈值q
  * 生成预测区间：模型给出预测值p，预测区间为[p-q,p+q]
  * 在论文中用来预测随机干扰引起的误差轨迹的范围
* 更紧致的概率保证：以往依赖于联合界等保守估计算法，系统被严苛控制，本文计算更精确的误差界，让系统的控制宽松点
  * 联合界：几个事件中至少发生一个的概率，不会超过它们各自概率的总和 $P(A or B)\leq P(A) + P(B)$
  * $P(总失败)\leq P(任务1失败)+\cdots+P(任务N失败)$
  * 为了让总失败率小于5%，需要保证每个子任务失败率小于 $\frac{5}{N}$ %，此时对每个子任务要求严苛，系统行为过于保守以至于找不到解
* 系统状态分解：
  * 没有随机干扰、可预测的标称部分
  * 由随机干扰引起的误差部分
  * 控制策略对应设计成标称控制和对干扰的反馈控制，问题转变成为误差轨迹范围提供一个高概率的边界
* 定义非符合性分数：衡量误差的不寻常程度，分越高则越偏离正常范围
* 利用STL公式将全局优化问题分解为子问题：从任务中心转变为智能体中心，完成全局任务等于每个智能体完成自己的任务
  * STL公式：形式化语言描述一个系统的行为随时间演变的规则
  * 子任务并非完全独立，比如智能体1和2都参与了任务2，则它们的决策必须相互协调，此时需要分布式算法
  * 每个智能体求解自己的子问题，然后将自己的计划和决策与协同任务的伙伴进行通信，再根据伙伴信息进行调
* 算法2：每个智能体计算自己计划的危险指数，对所有任务的安全分数中取最小值，如果是小组中最危险的就重新规划，如果不是最低的就保持

[**Large Language Model Agent: A Survey on Methodology, Applications and Challenges**](https://www.alphaxiv.org/abs/2503.21460) 2025.4

[**PLAN-AND-ACT: Improving Planning of Agents for Long-Horizon Tasks**](https://www.alphaxiv.org/abs/2503.09572) 2025.4

* 提升LLM执行复杂的长时长任务（如网页导航）的能力
* 规划与执行分离
  * PLANNER规划器：将用户指令分解成一个结构化的、分步骤地宏观计划
  * EXECUTOR执行器：接收规划器的每一步计划，并转化为特定环境中可执行的动作（点击、输入文字）
* 动态规划调整：每执行一步操作，规划器会根据环境反馈重新评估并更新后续计划
* 数据合成的逆向工程：对一个大模型成功执行任务的轨迹生成对应的高层次计划（好比对一张好看的照片，让模型生成对应的指令，再通过修改指令来修改生成的图片）

[**Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution**](https://www.alphaxiv.org/abs/2504.16563) 2025.4

[**A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents**](https://www.alphaxiv.org/abs/2504.14650) 2025.4

[**ReloPush: Multi-object Rearrangement in Confined Spaces with a Nonholonomic Mobile Robot Pusher**](https://www.alphaxiv.org/abs/2409.18231) 2025.3

[**RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints**](https://www.alphaxiv.org/abs/2503.16408) 2025.3

[**REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation**](https://www.alphaxiv.org/abs/2503.22122) 2025.3

[**EMOS: EMBODIMENT-AWARE HETEROGENEOUS MULTI-ROBOT OPERATING SYSTEM WITH LLM AGENTS**](https://www.alphaxiv.org/abs/2410.22662) 2025.2

[**Multi-Agent Collaboration Mechanisms: A Survey of LLMs**](https://www.alphaxiv.org/abs/2501.06322) 2025.1

[**Multi-Agent Motion Planning with B´ezier Curve Optimization under Kinodynamic Constraints**](https://www.alphaxiv.org/abs/2311.14145) 2024.3

[**Asynchronous Task Plan Refinement for Multi-Robot Task and Motion Planning**](https://www.alphaxiv.org/abs/2309.08897) 2023.9
