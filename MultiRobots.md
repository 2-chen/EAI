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
 
  * [**CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks**](https://www.alphaxiv.org/abs/2509.14380) 2025.10

* 大语言模型 (LLMs) 和视觉语言模型 (VLMs) 作为自动化教练，教会多机器人系统完成复杂的协作任务
* 利用LLM将复杂目标分解为子任务，用LLM生成奖励函数
* VLM闭环：
  * 评估：观察机器人执行任务的情况
  * 反馈：若失败会分析原因（奖励函数的学习曲线），生成建议
  * 优化：LLM接受建议并修改奖励函数
* 序贯训练：前一个子任务训练好的策略权重会用作下一个更难子任务的初始权重
