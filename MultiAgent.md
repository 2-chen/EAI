[**Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation**](https://www.alphaxiv.org/abs/2507.18224) [open](https://github.com/Shiy-Li/ARG-Designer) 2025.11

* 自动设计多智能体系统的通信拓扑结构，根据用户输入从零开始动态地组建一支团队并设计沟通流程
* 现有方法：
  * 组合冗余：从一个预定义的、包含所有可能角色和密集连接的模板图开始，通过剪枝或重加权来生成特定任务的图，导致初始计算成本高，且往往会保留不必要的代理或连接，造成资源浪费
  * 扩展受限：由于模板固定，如果引入新的agent，需要重新定义整个模板并重新训练
* 可扩展的角色池：基于度量学习的角色选择机制，使得模型可以从未见过的、扩展的角色池中选择agent
  * 基于度量学习的角色选择机制：学习生成一个意图向量，将其与所有候选agent的向量表示进行比对
  * 将所有可选角色转化为固定维度的角色：每个角色有一段自然语言描述，通过编码、提取特征、降维与规范化得到固定维度向量，因为是基于语义嵌入的，所以功能相似的角色在向量空间中的距离也会很近
* 模型架构
  * 编码用户的任务查询query和候选agent的描述
  * 节点生成：使用GRU_node决定下一个加入团队的agent，GRU_node是一个循环神经网络模块，维护着节点生成的历史状态，在第i步接收之前的上下文信息（如已存在的agent），更新隐藏状态，并基于这个状态决定添加哪个新节点
  * 边生成：选定新节点后，使用另一个GRU_node遍历之前已存在的所有节点，逐个判断是否需要建立从旧节点到新节点的连接，通过MLP处理隐藏状态进行判断
* 训练：使用复杂的、资源丰富的图结构进行训练，教会模型如何生成正确的图（也是数据驱动）
* 微调：使用经过剪枝的、极简的图结构进行微调，教会模型在保证正确率的前提下生成更稀疏的拓扑
* 传统方法会受原始模板图节点数的限制，比如模板图是5个节点，就剪不出6个节点的图，而ARG可以生成6个节点的图

*想法*：这个数据集是否有通用性，或者模型是否有通用性，直接调用来服务coffee demo

[**Latent Collaboration in Multi-Agent Systems**](https://www.alphaxiv.org/abs/2511.20639) [open](https://github.com/Gen-Verse/LatentMAS) 2025.11

* LatentMAS 实现了端到端的协作，智能体之间的思考和交流完全在连续的向量空间中进行，直到最后才解码成文本
* 当 Agent A 完成思考后，它的 KV Cache（包含了输入信息和它刚刚生成的隐思维）会被直接拼接到 Agent B 的 Transformer 层中
* 输入-输出对齐：为了让模型能理解前一步生成的“隐藏状态”，设计了一个线性对齐机制，引入一个一个线性投影矩阵 $W_a$ ， $e=hW_a$ ，其中h是输出的隐藏状态，e是对齐后的输入向量， $W_a$ 由最小二乘法计算，目的是将输出空间映射回有效的输入嵌入空间
* 输出KV Cache：在 PyTorch 或 HuggingFace Transformers 库中，模型的前向传播（Forward Pass）不仅会返回 logits（用于生成文本），还会返回 past_key_values（即 KV Cache），调用底层model.forward()
* KV Cache 的数据体积（显存占用/传输带宽）远大于纯文本字符串，牺牲了显存，换取高效计算、无损信息
* 适合planner、critic等智能体接力，中途不需要对外界输出，最多也就适合中央大脑传输给机器人的时候

*想法*：不适合多机控制的传输，多机传输带宽有限，要求数据信息密度高，文本token的信息密度比KV Cache高，且可以中途进行输出，也可以人为进行语言干预。最好先单独计算后再传输集成度更高的信息，而KV Cache计算完成度不高

[**Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning**](https://www.alphaxiv.org/abs/2511.10037) 2025.11

* 设计输出为全局 DAG 规划：模型不是生成线性的步骤，而是一次性生成一个有向无环图 (DAG)，其中节点代表工具，边代表依赖关系。这使得系统能够识别并行执行的机会，并从全局视角优化流程

[**Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning**](https://www.alphaxiv.org/abs/2511.03591v1) 2025.11

[**VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning**](https://www.alphaxiv.org/abs/2506.09049) [open](https://faceong.github.io/VIKI-R/) 2025.10

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
* 步数惩罚：奖励动作序列短的，结果性能更强，和正式动作序列长度更相似
* 和人类专家对比

[**Global-State-Free Obstacle Avoidance for Quadrotor Control in Air-Ground Cooperation**](https://www.alphaxiv.org/abs/2510.24315) 2025.10

[**Few-Shot Demonstration-Driven Task Coordination and Trajectory Execution for Multi-Robot Systems**](https://www.alphaxiv.org/abs/2510.15686) 2025.10

* 提出了一种少样本学习框架，将多机器人任务学习在结构上进行了解耦，旨在解决多机器人系统的协调与轨迹执行问题
* 时空解耦：分别学习任务规划和具体执行
  * 时间上的动作序列协调：学习谁在什么时间应该做什么事
  * 空间上的运动轨迹生成：学习机器人在执行具体动作时如何移动
* 基于谱聚类的关键关系提取：识别和提炼出机器人之间、机器人与物体之间最重要的交互关系
  * 谱聚类：将数据点看作图中的节点，试图找到一种切割图的方式，使得被切开的各个子图内部连接紧密，而子图间连接稀疏，交互频率影响每条边的权重
* 通过时间图网络（TGN）学习任务无关的时间序列
  * 学习高级的动作顺序和协调逻辑
  * 将机器人、目标等实体看作图的节点，利用图注意力网络(GATs)捕捉每一个关键时间点上各个实体之间的空间关系和状态
  * 将由GATs编码的、代表系统状态的图快照序列，输入到一个门控循环单元(GRU)，GRU负责学习这些状态随时间演变的规律
    * GRU：一种循环神经网络的变体，GRU有一个记忆单元，当读取序列中下一个元素时，结合上一时间步的隐藏状态（记忆）来更新自己的理解
    * 更新门：决定多少过去的记忆应该被保留到当前
    * 重置门：决定多少过去的记忆需要被忽略或重置
  * 最终TGN以自回归的方式根据当前和过去的状态，预测出下一步整个系统应该执行的组合动作
* 并利用高斯过程（GP）进行连续的空间轨迹建模
  * 负责生成具体、平滑的机器人运动路径
  * 从人类演示的轨迹中学习一系列标准化的运动基元，这些轨迹先被归一化处理，消除了尺寸、旋转和位置的影响
  * 高斯过程学习的是从标准化的时间进度（0%到100%）到标准化坐标的映射关系
    * 将轨迹标准化，让模型学习通用的形状而不是具体的路径
  * 当TGN模块给出一个移动指令后（例如，机器人A移动到位置B），GP就能根据新的起点和终点，将学到的标准轨迹进行缩放、旋转和平移，生成一条符合当前场景要求、几何形状与演示相似的平滑轨迹

*想法*：对于一个原始数据，应该分别学习其内部不同特征，而不是全部混在一起学习，容易互相干扰，分层思想，也可以交由不同的智能体学习

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

* LLMs和VLMs作为自动化教练，教会多机器人系统完成复杂的协作任务（将LLM和VLM的知识蒸馏到机器人上）
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
  * KL散度惩罚：惩罚当前模型和原始预训练模型的差异，防止过拟合（强化学习中常用，把原始模型作为参考模型）
  * 优势归一化：减去平均值除以标准差，保持模型更新的稳定，优势信号更平滑
  * 限制单次参数更新的幅度，防止因为一次高奖励进行破坏式更新

*想法*：完全通过多智能体内部交互实现迭代，没有引入外部数据和反馈，本质上是对内部数据进行更深层次的耦合，把方案、评估、评分对应的数据深度耦合，非常依赖本身的数据规模。Transformer是对所有数据进行耦合，这是更高层次、更有选择性的耦合，类似于特调，所以是否还可以引入更多层次的耦合，提高面对特定任务的质量？（这就是少样本学习？）

[**Multi-Actor Multi-Critic Deep Deterministic Reinforcement Learning with a Novel Q-Ensemble Method**](https://www.alphaxiv.org/abs/2510.01083v1) 2025.10

* 同时训练多个“执行家”和多个“评论家”（多就是好，数量多或者层数多）
* 双目标选择执行动作：预期回报和创意性
* 得分取所有评论家评分的中位数
* actor学习时每一轮让不同critic评分，以前是一个actor和一个critic配对

[**RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning**](https://www.alphaxiv.org/abs/2509.05397) 2025.9

[**A Framework for Scalable Heterogeneous Multi-Agent Adversarial Reinforcement Learning in IsaacLab**](https://directlab.github.io/IsaacLab-HARL/papers/paper-harl-a.html) [open](https://directlab.github.io/IsaacLab-HARL/papers/paper-harl-a.html) 2025.9

* 在高保真物理仿真中进行大规模、异构、对抗性多智能体训练
* 将 HAPPO (Heterogeneous Agent PPO) 算法扩展到竞争环境，引入了分队评论家机制，为每个团队配置独立的评论家网络
* 新的基准环境：相扑、足球、空地拦截
* 零缓冲课程学习：一种处理动态观测空间的课程学习方法，使得智能体可以从简单任务平滑过渡到复杂的对抗任务
  * 在观测空间中预留零填充的特征位，框架可以在课程后期无缝地引入新的观测信息，而无需重新训练整个模型
* 不同机器人协同对抗另一组不同机器人
* 设计一个团队共享奖励，把对手作为一个整体，队友作为一个整体，比如奖励函数中奖励对手倒下的人数，惩罚队友倒下的人数

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

[**Heterogeneous multi-agent reinforcement learning for zero-shot scalable collaboration**](https://directlab.github.io/IsaacLab-HARL/papers/paper-harl.html) open 2025.6

[**EMBODIEDBENCH: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents**](https://embodiedbench.github.io/) open 2025.6

[**Multi-agent embodied ai: Advances and future directions**](https://www.alphaxiv.org/abs/2505.05108) 2025.6

* 多智能体系统的复杂性：通信、协调、冲突解决和信任分配
* 多智能体强化学习的挑战：异步协作（智能体决策时间不一致），如何根据团队整体表现来评估个体智能体的贡献
  * 多智能体异步学习：不同智能体决策和动作是错开的，智能体2在做出决策 $u_2$ 时，可能还没有收到智能体1执行 $u_1$ 后的最新消息，这给协同带来挑战，
* 新算法设计：现有算法多在理想化的模拟环境中开发，需要设计更能适应真实世界（如传感器噪声、延迟反馈）的稳健算法
* 通用框架：能处理多任务、多场景的通用多智能体具身AI框架
* 评估与验证：缺乏标准化、可复现且贴近现实的评估体系，尤其是在安全性和可靠性方面
* IRL逆强化学习：从专家轨迹中反推出其背后的奖励函数，一旦学到了奖励函数就可以用标准的强化学习训练一个最优策略，泛化性好
* GAIL生成对抗模仿学习：生成器产生和专家相似的行为轨迹，判别器区分那些轨迹来自专家哪些来自智能体，使得智能体策略趋近专家行为
* Mamba：序列模型框架，基于状态空间模型SSM，通过选择机制实现对序列数据的线性时间复杂度的处理，有效捕捉长距离依赖（？
* RRT算法？
* 轨迹优化：无碰撞和利用碰撞，在某些狭窄或动态的环境中，与可变形的障碍物（如软布）发生轻微碰撞可能比完全规避它花费的时间更短
* 世界模型：
  * 针对具身智能体：模型接受每个智能体的第一人称视角观测和联合动作，预测未来的第一人称视角观测

[**Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs**](https://danielchyeh.github.io/All-Angles-Bench/) open 2025.4

* 多视角理解能力

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

[****]

[**COHERENT: Collaboration of Heterogeneous Multi-Robot System with Large Language Models**](https://www.alphaxiv.org/abs/2409.15146) open 2025.3

* COHERENT 框架：基于 LLM 的集中式分层架构，专门用于异构机器人的任务分配和协作
* PEFA 机制：Proposal-Execution-Feedback-Adjustment（提议-执行-反馈-调整） 闭环机制，使得系统能够根据执行结果动态调整计划
  * 任务分配者收到反馈后，结合历史信息更新认知。如果任务失败或仅部分完成，它会重新生成新的提议（Adjustment），直到最终目标达成
* 自反思反馈：当底层机器人执行失败时，它不是简单报错，而是会分析原因（如“无法到达”、“能力不足”）反馈给上层，上层再重新规划。这比单纯的单向指令下达要鲁棒得多（要利用每个机器人自身的计算能力，先自己处理再压缩成高级语义进行通信，简单高效很多）

[**MultiAgentBench : Evaluating the Collaboration and Competition of LLM agents**](https://www.alphaxiv.org/abs/2503.01935) open 2025.3

* 衡量多智能体在复杂协作和竞争场景下的综合表现
* 六种场景的基准测试
  * 协作任务：智能体有共同目标，如共同撰写研究计划、游戏世界里合作建造、集体调试代码
  * 对抗任务：智能体有冲突目标，需要通过策略、欺骗和谈判来获胜
* 多维评估指标：既看结果，也看过程
  * 任务完成度：设定动态里程碑
  * 协作质量：需要设计一套详细的评分标准，也就是prompt
    * 规划分数：计划的合理性、可行性、创新性
    * 沟通分数：沟通的清晰度、效率、贡献
* 灵活的协调框架：可以使用这个框架测试不同团队的组织方式和沟通策略
  * 多样的协作协议：模拟不同团队的管理模式，如中心化的星形和树形结构，去中心化的网状和链式结构
  * 规划策略：除了思维链和小组讨论，还提出了认知自进化规划，根据过去经验（期望与实际结果的对比）来动态调整未来的计划

[**ReloPush: Multi-object Rearrangement in Confined Spaces with a Nonholonomic Mobile Robot Pusher**](https://www.alphaxiv.org/abs/2409.18231) 2025.3

* 让一个有非完整约束的移动机器人（不能横着走），在一个狭窄且堆满物体的空间里，通过推的方式，将多个物体重新排列到指定的目标位置
  * 运动学约束：转弯半径有限，不能随心地移动
  * 物理约束：为了稳定地推动物体而不让它滑走或翻倒，转弯速度和加速度受到限制
  * 复杂交互：一个物体会挡住另一个物体的路径，需要策略性地移开挡路的物体，这种被称为非单调重排问题
* 推行-可达性图(PT-graph)：节点代表机器人可以开始推物体的姿态，边则代表一个推动姿态到另一个姿态的满足所有运动学和物理约束的最优路径（杜宾斯曲线？），将复杂的几何、运动学和稳定性约束都编码到图结构中，使规划更高效
* 构建PT-graph：
  * 系统先分析场景中所有物体的起始和目标位置，为每个位置生成几个有效的推动姿态作为图节点，
  * 计算任意两节点之间是否存在一条无碰撞、满足稳定推动条件的杜宾斯曲线，存在就连边，权重是路径长度
  * 构建好后搜索成本最低的重排方案(Dijkstra)
* 预重定位：当机器人发现从物体当前位置推到目标位置的路径会出界时，它不会放弃，而是先小范围移动这个物体，改变起始推动姿态，从而找到一条完全在边界内的可行路径

*想法*：人为设计一个图结构来表示数据或环境，可以让模型更好地学习

[**RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints**](https://www.alphaxiv.org/abs/2503.16408) 2025.3

* 如何自动、安全、高效地生成用于训练的数据
* 组合式约束：将多机器人协作中复杂的、不成文的规则形式化和结构化的方法
  * 逻辑约束：定义机器人与物体交互的正确方式，比如在拍照任务中应该抓住握柄而不是镜头
  * 空间约束：确保机器人在共享的物理空间中不会互相碰撞，或者碰到其他物体
  * 时间约束：管理机器人动作的执行时序和同步性，例如先打开盖子，才能把食物放进去，两个机器人可以同时移动，只要路径不冲突
* RoboChecker检查器：接受来自LLM大脑的文本约束，转化为机器人可以理解和执行的约束接口
  * 逻辑约束：检查末端执行器是否在标注好的正确交互点上
  * 空间约束：深度摄像头或算法建立场景的3D占用地图，检测运动轨迹是否进入其他物体的占用空间
  * 时间约束：分析轨迹在时间维度上的占用情况，判断是否存在时序错误或不合理等待
* 安全性：直接让模型生成动作，容易出现互相碰撞的行为
* 效率问题：自动生成数据，依靠LLM生成结构化数据，重点在于设计结构

[**REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation**](https://www.alphaxiv.org/abs/2503.22122) 2025.3

[**EMOS: EMBODIMENT-AWARE HETEROGENEOUS MULTI-ROBOT OPERATING SYSTEM WITH LLM AGENTS**](https://www.alphaxiv.org/abs/2410.22662) 2025.2

[**Multi-Agent Collaboration Mechanisms: A Survey of LLMs**](https://www.alphaxiv.org/abs/2501.06322) 2025.1

[**Large Language Model based Multi-Agents: A Survey of Progress and Challenges**](https://www.alphaxiv.org/abs/2402.01680) 2024.4

[**Multi-Agent Motion Planning with B´ezier Curve Optimization under Kinodynamic Constraints**](https://www.alphaxiv.org/abs/2311.14145) 2024.3

[**Heterogeneous-Agent Reinforcement Learning**](https://www.alphaxiv.org/abs/2304.09870) open 2023.12

[**Asynchronous Task Plan Refinement for Multi-Robot Task and Motion Planning**](https://www.alphaxiv.org/abs/2309.08897) 2023.9

[**RoCo: Dialectic Multi-Robot Collaboration with Large Language Models**](https://www.alphaxiv.org/abs/2307.04738) [open](https://project-roco.github.io/) 2023.7

* 对话式协作，赋予每个机器人一个身份，让他们通过自然语言对话来商量策略，提高可解释性
* LLM生成具体的任务空间路径点，作为底层运动规划器的提示（VIKI-R的L3轨迹感知）
* 环境反馈闭环：物理环境到语言环境的反馈机制，若LLM生成的计划在物理上不可行，环境会生成文本形式的错误反馈，让LLM进行上下文学习并修正计划


