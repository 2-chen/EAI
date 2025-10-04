[**World Models**](https://www.alphaxiv.org/abs/1803.10122)
* 智能体先无监督学习环境的生成模型（世界模型），然后在该模型产生的“幻觉”或“梦境”中进行训练，最后将学到的策略迁移回真实环境中
* 视觉模型（Vision Model）：将环境提供的高维观察压缩成低维的紧凑表示，称为潜在向量 $z$
* 记忆模型（Memory Model）：给定当前的潜在向量 $z_t$和动作 $a_t$，预测 $z_{t+1}$的概率分布，RNN 的隐藏状态 $h_t$封装了时间的动态信息，看作是“记忆”
* 控制器（Controller Model）：接收来自视觉模型的 $z_t$和来自记忆模型的 $h_t$，然后输出要在环境中执行的动作 $a_t$

[**A Survey: Learning Embodied Intelligence from Physical Simulators and World Models**](https://arxiv.org/pdf/2507.00917)

