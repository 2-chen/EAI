[**Mathematics of multi-agent learning systems at the interface of game theory and artificial intelligence**](https://arxiv.org/pdf/2403.07017?)

* 将演化博弈论和多智能体学习结合起来
* 演化博弈论中个体可以在博弈后复制对方的策略
* 智能体学习智能体得到奖励，并不知道交互对象的策略，往往其交互对象是环境也没有策略\
* 若要结合演化博弈论，需要假定在智能体交互的场景下，且智能体可以感知交互对象的策略，进而调整自己的策略权重，复制或学习对方的策略权重

[**Evolutionary game selection creates cooperative environments**](https://arxiv.org/pdf/2311.11128)

* 考虑了博弈的变化，不是固定的博弈类型
* 博弈选择会创造一个更有利于合作的环境，不利于合作的博弈会被淘汰，没有玩家会想参与类似囚徒困境的劣质博弈，策略和博弈组合不够吸引人
* 玩家会学习榜样玩家（总收益高）的博弈类型和博弈策略，学习概率和收益差成正比
* 博弈阶段和学习阶段是独立的，不会直接学习博弈对象的策略

*想法*

不止学习榜样，遇到收益比自己低的玩家，要主动摒弃他的博弈类型和策略，可以更快地学习？
