# 经典论文
**Deep Unsupervised Learning using Nonequilibrium Thermodynamics**

自定义添加噪声的函数（Gaussian函数），预设去除噪声和添加噪声为同一个函数，所以可以学习参数进行去噪

在知道初始状态的情况下，$q(x^{(t-1)}|x^{(t)},x^{(0)})$可以通过贝叶斯定理精确计算，其均值和方差可以用$x^{(t)}$和$x^{(0)}$表示

训练过程：
* 取初始状态$x^{(0)}$，模拟加噪得到$x^{(t)}$
* 输入$x^{(t)}$进行去噪
* 去噪结果和$x^{(0)}$对比
* 计算损失通过反向传播算法调整神经网络参数

**Denoising Diffusion Probabilistic Models**

**Improved Denoising Diffusion Probabilistic Models**
