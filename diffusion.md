# 经典论文

[**Optical generative models**](https://www.alphaxiv.org/abs/2410.17970) 2025.8

* 光学生成模型：利用光在物理空间中传播和衍射的自然过程来完成复杂的计算
* 空间光调制器(Spatial Light Modulator, SLM)：每个像素点是光的相位
* 光学解码器：光穿过SLM后相位上被雕刻种子的信息，再通过衍射层发生复杂的衍射和干涉，相当于矩阵运算，生成最终图像
* 乘法：振幅乘以透光率
* 加法：源平面所有子波的叠加


[**Improved Denoising Diffusion Probabilistic Models**](https://www.alphaxiv.org/abs/2102.09672) 2021.2

同时学习噪声的方差，原方法只学习均值

$$
L_{hybrid}=L_{simple}+\lambda L_{vlb}
$$
$$
L_{vlb}:=L_0+L_1+\cdots +L_{T-1}+L_T
$$
$$
L_0:=-\text{log} p_\theta (x_0|x_1)
$$
$$
L_{t-1}:=D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))
$$
$$
L_T:=D_{KL}(q(x_T|x_0)||p(x_T))
$$

预测方差：给出下界 $\beta_t$ 和上界 $\tilde{\beta}_t$ ，网络只需要学习 $v$ ，得到插值作为方差

$$
\Sigma_\theta (x_t,t)=\text{exp}(v\text{log}\beta _t+(1-v)\text{log}\tilde{\beta}_t)
$$


[**Denoising Diffusion Probabilistic Models**](https://www.alphaxiv.org/abs/2006.11239) 2020.11

单独学习每一步添加的噪声，实现每一步单独去噪

使用U-net架构神经网络，在上采样时连接同等尺寸的下采样特征图（特征通道叠加），使得在恢复图像细节时同时利用局部精细特征和全局语境特征

[**Deep Unsupervised Learning using Nonequilibrium Thermodynamics**](https://www.alphaxiv.org/abs/1503.03585) 2015.11

自定义添加噪声的函数（Gaussian函数），预设去除噪声和添加噪声为同一个函数，所以可以学习参数进行去噪

在知道初始状态的情况下， $q(x^{(t-1)}|x^{(t)},x^{(0)})$ 可以通过贝叶斯定理精确计算，其均值和方差可以用 $x^{(t)}$ 和 $x^{(0)}$ 表示

训练过程：
* 取初始状态 $x^{(0)}$ ，模拟加噪得到 $x^{(t)}$

* 输入 $x^{(t)}$ 进行去噪
* 去噪结果和 $x^{(0)}$ 对比
* 计算损失通过反向传播算法调整神经网络参数
