[**GTALIGN: GAME-THEORETIC ALIGNMENT OF LLM ASSISTANTS FOR MUTUAL WELFARE**](https://www.alphaxiv.org/abs/2510.08872) 2025.10

* 将用户和LLM的互动形式化为一个策略游戏，传统对齐方法中LLM为了追求自身奖励最大化生成冗长的答案，反而损害了用户体验，导致双方都不是最优的
* GTAlign让LLM生成回答时内部进行一个明确的博弈论推理过程，在思维链中构建一个收益矩阵评估不同策略对用户和LLM的价值，选择一个最大化收益的策略
* 帕累托最优(Pareto Optimality)：不可能在不损害任何一方利益的情况下，让某一方的利益变得更好，任何调整都是零和的
* 构建推理链：模型输出通过prompt被设计为包含四个部分的结构化格式
  * thinking：初步分析
  * payoff：JSON格式生成博弈矩阵
  * analysis：分析收益矩阵，找出最大化共同福利的策略
  * response：根据分析结果给出最终答案
 * 人工合成遵循四段式博弈论推理格式的高质量数据集，让模型初步学习结构化思考方式，系统根据回答给出福利评分，合成共同福利作为奖励信号
 * 柯布-道格拉斯函数 $W_{mutual}=\sqrt{U\cdot L}$ ，乘法性质使得任何一方福利趋于0都会使得共同福利趋于0，且有边际效益递减，避免两方福利差太多
 * 评估价值依据：答案质量，交互成本，需求满足度，计算成本，任务成功率，格式对齐，安全性
 * 进行调控时，在模型生成<payoff>矩阵后暂停，外部修改这个矩阵中的数值，然后再将修改后的矩阵注入回去，让模型继续完成<analysis>和<response>的生成
 * 控制模型的后端系统可以监控LLM的生成流，检测到停止标记</payoff>时会暂停对模型后续token请求，将文本字符串中旧矩阵替换为人为修改的矩阵，得到新的上下文prompt，让模型从暂停点继续生成，对模型而言不会觉得有中断，本质上是修改prompt

[**A Game Theory-Reinforcement Learning Approach to Cooperation for UAVs**](https://www.alphaxiv.org/private/448f0c11-d285-48cc-bc55-ce226dcd1214) 2025.6

* 将博弈论和强化学习结合，解决动态环境下无人机集群的协作问题
* PGG：每个个体都可以选择是否为一个公共项目“投资”，所有投资乘以一个增益系数，然后平均分配给所有成员，最理性的“自私”策略就是不投资，等着分享别人的成果，最终结果是没人投资，集体利益为零。
* 非对称环境反馈：“合作者”和“背叛者”获得的收益不同，背叛者的收益乘数 $r_d$ 是一个固定值，而合作者的收益乘数 $r_c$ 是一个可变值
* 将无人机集群的协作问题建模为带有“非对称环境反馈”的公共物品博弈（Public Goods Game, PGG），考虑了资源限制（例如通信带宽、能源），非对称意味着合作者和背叛者（不分享信息者）的收益计算方式不同，合作者的收益乘数会根据环境动态变化
* 将描述群体策略演化的博弈论“复制动态方程”与强化学习的“选择和变异机制”在数学上统一起来
* 引入一种名为频率调整Q学习，使得无人机能够在未知和动态变化的环境中自主学习最优策略，解决了传统博弈论模型通常是静态的，以及传统强化学习模型难以描述复杂多智能体策略互动的问题
* 强化学习的Softmax探索策略

$$
x_t^j = \frac{e^{Q_t^j/\tau}}{\sum_{l=1}^n e^{Q_t^l/\tau}}
$$

其中 $x_t^j$ 表示第 $t$ 轮选择动作 $a_j$ 的概率， $Q_t^j$ 表示第 $t$ 轮动作 $a_j$ 的价值， $\tau$ 是温度因子
* 合作者和背叛者的利益分别为，其中合作者有成本

$$
\begin{cases}
 \pi_C^i(t) = c \times \left( \frac{(m_i(t)+1) \times r}{d_i+1} - 1 \right) \\
 \\
 \pi_D^i(t) = c \times \left( \frac{m_i(t) \times r}{d_i+1} \right)
\end{cases}
$$

* 分别给合作者和背叛者乘上不同收益率，用来调整合作和背叛的意愿

$$
\begin{cases}
 \pi_C^i(t) = c \times \left( \frac{(m_i(t)+1) \times r_c}{d_i+1} - 1 \right) \\
 \\
 \pi_D^i(t) = c \times \left( \frac{m_i(t) \times r_d}{d_i+1} \right)
\end{cases}
$$



[**Evolutionary game selection creates cooperative environments**](https://arxiv.org/pdf/2311.11128) 2024.7

* 考虑了博弈的变化，不是固定的博弈类型
* 博弈选择会创造一个更有利于合作的环境，不利于合作的博弈会被淘汰，没有玩家会想参与类似囚徒困境的劣质博弈，策略和博弈组合不够吸引人
* 玩家会学习榜样玩家（总收益高）的博弈类型和博弈策略，学习概率和收益差成正比
* 博弈阶段和学习阶段是独立的，不会直接学习博弈对象的策略

*想法*

不止学习榜样，遇到收益比自己低的玩家，要主动摒弃他的博弈类型和策略，可以更快地学习？

[**Mathematics of multi-agent learning systems at the interface of game theory and artificial intelligence**](https://arxiv.org/pdf/2403.07017?) 2024.3

* 将演化博弈论和多智能体学习结合起来
* 演化博弈论中个体可以在博弈后复制对方的策略
* 智能体学习智能体得到奖励，并不知道交互对象的策略，往往其交互对象是环境也没有策略
* 若要结合演化博弈论，需要假定在智能体交互的场景下，且智能体可以感知交互对象的策略，进而调整自己的策略权重，复制或学习对方的策略权重

[**ALYMPICS: LLM Agents meet Game Theory Exploring Strategic Decision-Making with AI Agents**](https://www.alphaxiv.org/abs/2311.03220) 2024.1

* 提供一个用于博弈的实验平台，大模型驱动参与者，可进行角色设定


