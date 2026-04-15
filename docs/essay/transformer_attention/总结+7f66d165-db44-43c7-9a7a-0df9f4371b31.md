## 类别1：Hebbain/STDP类

Hebbain 是只要在这个激发附近就增加这个权重；STDP 在Hebbain的基础上升华→与因果时间有关系，在spike之前的加权重/spike之后的减权重

- **(1) supervised Hebbian**：

    - $\text{pre-post timing→weight update}$

- **(3) ReSuMe**

    - actual-desire

    - STDP+anti-STDP

    - 很经典的STDP的方法，是通过正确增加、错误减少的方式逼近正确结果

- **(6)Chronotron version 2 / I-learning**

    - 这个版本的**Chronotron**就是经过ReSuMe启发而得的，也就是：

        - 和上面的基本相同就是使用更加精细的双指数核，而不是 ReSuMe 那种单指数 STDP 窗口

- **(7) SWAT**

    - STDP+BCM

    - BCM作用是控制学习的强度

    - 但是这只能处理rate coding，不是精准的timing

- **(8) DL-ReSuMe**

    - 就是相对于STDP多学了一个synaptic delay

特点：

- 基于生物机制

- 局部更新

- 不严格优化目标

---

## 类别2：统计/优化类（目标函数驱动）

- (2) statistical methods

    - $maximaize(P(desired\text{ } spikes))$

---

## 类别3：信号变化/连续优化方法

- **(4) SPAN**
这个最典型。它把 spike train 转成 analogue signals，然后在连续域里做学习。
所以它更像：

    - signal-domain learning

    - continuous optimization

    - Widrow-Hoff 风格的函数逼近

    不是纯 STDP 法，也不是统计似然法。

---

## 类别4：显示误差函数/梯度近似

- 明确写出输出 spike train 与目标 spike train 之间的误差

- 试图通过梯度或近似梯度来最小化这个误差

- **(5) Chronotron version 1 / E-learning**
这个比 ReSuMe 更“优化派”一些，因为它显式定义：

    - insertion

    - deletion

    - shift

    这些 spike train 结构性误差，然后尝试做

    $Δw_j​=−\frac{​∂E}{∂w_j}​$

---

## 类别5：Reservoir/system-level methods

借助一个动态系统或 reservoir，把时间模式变成内部状态，再由 readout 处理。

- **(9) LSM**
LSM 本体不是在训练 reservoir 产生某个目标 spike train，而是：

    - reservoir 固定

    - 读出层训练

    它利用递归动态和短时记忆来表达时间信息。



- **(10) polychronous groups + LSM**
这还是 reservoir/LSM 路线的变体，只是更强调：

    - STDP + conduction delays

    - reservoir 内出现 polychronous groups

    - readout 通过 delay 学习去捕捉不同类对应的 polychronous group

    它不是前面那种“单神经元目标 spike train 学习”的标准套路，而是更偏系统动力学 / reservoir classification。

