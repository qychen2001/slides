---
theme: seriph
background: https://cover.sli.dev
transition: slide-left
---

# Deepseek 到底强在哪？

## 从原理到展望

陈启源

2025/03/17

---
layout: two-cols-header
---

# 闲聊

::left::

<v-click>

## 爆火前夕的随口一说

<img src="/wechat.jpg" style="height: 400px;margin-top:10px" >

</v-click>

::right::

<v-click>

## 错过最大的机会

<img src="/ds.jpg" style="height: 400px;margin-top:10px">

</v-click>

---
layout: section
---

# 从PPO到GRPO

原理概览

---

# 大语言模型（LLM）的训练流程

<v-click>

LLM 的训练分为两个主要阶段：

</v-click>

<v-click>

1. **预训练（Pre-training）**：
   - 通过大规模数据进行"下一词预测"任务。
   - 目标是让模型学会语言的基本结构和知识。

</v-click>

<v-click>

2. **后训练（Post-training）**：
   - 进一步提升模型的推理能力和与人类偏好对齐。
   - 包括 **监督微调（SFT）** 和 **基于人类反馈的强化学习（RLHF）**。

</v-click>

---

# 后训练的详细流程

<v-click>

后训练分为两个阶段：

</v-click>

<v-click>

1. **监督微调（SFT）**：
   - 使用少量高质量的 **专家推理数据**（如指令遵循、问答、思维链数据）。
   - 目标是让模型模仿专家的推理能力。

</v-click>

<v-click>

2. **基于人类反馈的强化学习（RLHF）**：
   - 通过人类反馈训练奖励模型，然后用强化学习微调模型。
   - 目标是让模型与人类偏好对齐。

</v-click>

---

# DeepSeek 的高效后训练

<v-click>

DeepSeek 的创新之处：

</v-click>

<v-click>

- **跳过 SFT 阶段**：
  - 直接对基础模型（DeepSeek V3）应用 RLHF。
  - **好处**：
    - 提升计算效率。
    - 避免人类策划数据的偏差。
  - **挑战**：
    - 需要一个非常强大的基础模型。

</v-click>

<v-click>

- **引入 GRPO**：
  - 替代传统的 PPO（近端策略优化）。
  - 减少内存和计算开销（约 50%）。

</v-click>

---

# RLHF的流程

<v-click>

让我们将RLHF的工作流程分解为几个步骤：

</v-click>

<v-click>

   - **步骤1**：对于每个提示，从模型中采样多个响应；
   - **步骤2**：人类根据质量对这些输出进行排序；
   - **步骤3**：训练一个**奖励模型**，以预测模型响应的人类偏好/排名；
   - **步骤4**：使用**RL**微调模型，以最大化奖励模型的评分。

</v-click>

<v-click>

正如我们所见，这个过程相对简单，包含两个可学习的组件，即**奖励模型**和**RL部分**。现在让我们更详细地探讨每个组件。

</v-click>

---

# 奖励模型

<v-click>

奖励模型的作用与训练

</v-click>

<v-click>

实际上，我们无法让人类对所有模型的输出进行排序。一个节省成本的方法是让标注人员对LLM输出的一小部分进行评分，然后<span v-mark.red="+1">**训练一个模型来预测这些标注者的偏好**</span>——这就是奖励模型的由来。

</v-click>

<v-click>

将可学习的奖励模型表示为$R_\phi$。给定一个提示$p$，LLM生成$N$个响应$\{r_1, r_2,...r_N\}$。然后，给定人类评分者更喜欢响应$r_i$而不是$r_j$，奖励模型通过最小化以下目标进行训练： 

$$ \begin{align}
\mathcal{L}(\phi) = -\log \sigma(R_\phi(p, r_i) - R_\phi(p, r_j)),
\end{align}
$$
其中$\sigma$表示sigmoid函数。

</v-click>

<v-click>

> **NOTE**：该目标源自**Bradley-Terry模型**，该模型定义了评分者更喜欢$r_i$而不是$r_j$的概率为：
> $$
> P(r_i \succ r_j) = \frac{\exp\big(R_\phi(p, r_i)\big)}{\exp\big(R_\phi(p, r_i)\big) + \exp\big(R_\phi(p, r_j)\big)}.
> $$
> 取该概率的负对数似然即得到损失$\mathcal{L}(\phi)$。

</v-click>

---

# RL算法：PPO

<v-click>

Deepseek背后的功臣——GRPO的基础

</v-click>

<v-click>

首先，一个高层次的概述。PPO代表近端策略优化，它需要以下组件：

</v-click>

<v-click>

- **策略（$\pi_\theta$）**：经过预训练/SFT的LLM；

</v-click>

<v-click>

- **奖励模型（$R_\phi$）**：一个已训练并冻结的模型，提供给定**完整响应**的标量奖励；

</v-click>

<v-click>

- **批评者（$V_\gamma$）**：也称为价值函数，它是一个可学习的模型，接收**部分响应**并预测标量奖励。

</v-click>

---

# RL算法：PPO

<v-click>

Deepseek背后的功臣——GRPO的基础

</v-click>

<v-click>

一旦我们了解了工作流程，每个组件的作用就会变得更加清晰，该工作流程包含五个阶段：

</v-click>

<v-click>

1. **生成响应**：LLM为给定提示生成多个响应；

</v-click>

<v-click>

2. **评分响应**：奖励模型为每个响应分配奖励；

</v-click>

<v-click>

3. **计算优势**：使用GAE计算优势（稍后会详细介绍，它用于训练LLM）；

</v-click>

<v-click>

4. **优化策略**：通过优化总目标更新LLM；

</v-click>

<v-click>

5. **更新批评者**：训练价值函数，使其更好地预测部分响应的奖励。

</v-click>

---

# 术语

状态与动作

<v-click>

> 我们将使用**状态**，表示为$s_t$，以及**动作**，表示为$a_t$。
> 注意，这里的下标$t$用于表示**token级别**的状态和动作；相反，之前我们定义提示$p$和响应$r_i$时，下标$i$用于表示**实例级别**的响应。

</v-click>

<v-click>

假设我们给LLM一个提示$p$。然后，LLM开始一个接一个地生成长度为$T$的响应$r_i$：

</v-click>

<v-click>

- $t=0$：我们的状态只是提示，即$s_0 = \{p\}$，第一个动作$a_0$是LLM生成的第一个词token；
- $t=1$：状态变为$s_1 = \{p, a_0\}$，因为LLM在生成下一个动作$a_1$时依赖于该状态；

</v-click>

<v-click>

...

- $t=T-1$：状态为$s_{T-1} = \{p, a_{0:T-2}\}$，LLM生成最终动作$a_{T-1}$。

</v-click>

<v-click>

将这次的输出与之前的输出合并，所有的动作串在一起形成一个响应，即$r_i = \{a_0, a_1,...a_{T-1}\}$。

</v-click>

---

# 广义优势估计（GAE）

定义了在状态$s_t$（即提示+已生成的词）下，**特定动作**$a_t$（即词）比策略将采取的**平均动作**好多少。

<v-click>

$$
\begin{align}
A_t = Q(s_t, a_t) - V(s_t)
\end{align}
$$

其中$Q(s_t, a_t)$是在状态$s_t$下采取<span v-mark.red>特定动作$a_t$</span>的预期奖励，$V(s_t)$是在状态$s_t$下策略采取<span v-mark.red>平均动作</span>的预期奖励。

</v-click>

<v-click>

估计这种优势有两种主要方法，每种方法都有其利弊，即：
1) **蒙特卡罗（MC）**：使用完整轨迹（即完整响应）的奖励。这种方法由于奖励稀疏性而具有高方差——从LLM中取足够样本来使用MC进行优化是非常昂贵的，但它的偏差较低，因为我们能够准确建模奖励；
2) **时间差分（TD）**：使用单步轨迹奖励（即衡量刚刚生成的词在提示下的好坏）。通过这样做，我们可以在token级别计算奖励，这显著减少了方差，但由于我们无法准确预测部分生成响应的最终奖励，偏差增加了。

</v-click>

<v-click>

**这就是GAE的用武之地！** 它通过多步TD来**平衡偏差和方差**。然而，回想一下我们之前提到的，奖励模型在响应不完整时会返回0：在没有知道生成一个词之前的奖励的情况下，我们如何计算TD？因此，我们引入了一个可以做到这一点的模型，我们称之为“批评者”。

</v-click>

---

# 批评者（价值函数）

批评者被训练为**仅根据部分状态预测最终奖励**，以便我们可以计算TD。


<v-click>

训练批评者$V_\gamma$非常简单：

给定一个部分状态$s_t$，我们希望预测给定完整状态$s_T = \{p, r\}$的奖励模型输出。批评者的目标可以写成：

$$
\begin{align}
L(\gamma) = \mathbb{E}_t \left[(V_\gamma(s_t) - \text{sg}(R_\phi(s_T)))^2\right],
\end{align}
$$

其中$\text{sg}$表示停止梯度操作。正如我们所看到的，批评者通过简单的L2损失对奖励模型的评分进行训练。

</v-click>

<v-click>

你可能会注意到，虽然奖励模型$R_\phi$在PPO之前训练并冻结，但批评者与LLM一起训练，尽管它的工作也是预测奖励。这是因为价值函数必须估计给定<span v-mark.red>**当前策略**</span>的部分响应的奖励；因此，它必须与LLM一起更新，以避免其预测变得过时和不一致。而这，就是他们所说的，RL中的**演员-批评者**。

</v-click>

---

# 回到GAE
有了批评者$V_\gamma$，我们现在有了一种预测部分状态奖励的方法。

<v-click>

$$
\begin{align}
A^{\text{GAE}}_K = \delta_0 + \lambda \delta_1 + \lambda^2 \delta_2 ... + (\lambda)^{K-1} \delta_{K-1} = \sum^{K-1}_{t=0} (\lambda)^t \delta_t,
\end{align}
$$

</v-click>


<v-click>

其中$K$表示TD步数，$K<T$。$\delta_t$表示步骤$t$的TD误差，计算公式为：

$$
\begin{align}
\delta_t = V_\gamma(s_{t+1}) - V_\gamma(s_t)
\end{align}
$$

简而言之，TD误差计算了一步的预期总奖励，$A_{K}^{\text{GAE}}$通过计算$K$步的聚合单步TD误差来估计优势。

</v-click>

<v-click>

<span v-mark.red>GAE方程中的$\lambda$控制方差和偏差之间的权衡</span>：当$\lambda =0$时，GAE退化为单步TD；当$\lambda=1$时，GAE变为MC。

在RLHF中，我们希望最大化这个优势项，从而<span v-mark.red>最大化LLM生成的每个token的奖励。</span>

</v-click>

---

# 整合在一起——PPO目标
PPO目标包含几个组件，即1）剪裁替代目标，2）熵奖励，3）KL惩罚。

## 1. 剪裁替代目标

这是我们要最大化$A_K^{\text{GAE}}$的地方，以便LLM预测的每个token都能最大化奖励（或者，根据之前对优势的定义，LLM预测的每个token应该比其平均预测好得多）。剪裁替代目标通过概率比$c_t(\pi_\theta)$限制策略更新：

$$
\begin{align}
L^{\text{clip}}(\theta) = \mathbb{E}_t \left[ \min(c_t(\pi_\theta)A^{GAE}_t, \text{clip}(c_t(\pi_\theta),1-\epsilon, 1+\epsilon)A^{GAE}_t)\right],
\end{align}
$$

其中$\epsilon$控制剪裁范围，$c_t(\pi_\theta)$是在给定累积状态$s_t$下预测特定token$a_t$的概率比，更新前后：

$$
\begin{align}
c_t(\pi_\theta) = \frac{\pi_\theta (a_t | s_t)}{\pi_{\theta_{\text{old}}} (a_t | s_t)}.
\end{align}
$$


你可以将剪裁视为一种防止过度自信的方法，如果没有剪裁，较大的$A_K^{\text{GAE}}$可能导致策略对某一动作过度投入。

---

# 整合在一起——PPO目标

PPO目标包含几个组件，即1）剪裁替代目标，2）熵奖励，3）KL惩罚。

## 2. KL散度惩罚
此外，我们有KL散度惩罚，防止当前策略$\theta$偏离我们进行微调的原始模型$\theta_{\text{orig}}$：
$$
\begin{align}
\text{KL}(\theta) = \mathbb{E}_{s_t} \left[ \mathbb{D}_{\text{KL}}(\pi_{\theta\text{orig}}(\cdot | s_t) || \pi_{\theta}(\cdot | s_t)) \right]
\end{align}
$$

KL散度简单地通过对序列和批次取平均来估计。

## 3. 熵奖励
熵奖励通过惩罚低熵来鼓励LLM生成的探索：

$$
\begin{align}
H(\theta) = - \mathbb{E}_{a_t} [\log \pi_\theta (a_t | s_t)].
\end{align}
$$

---

# 最终，PPO目标

有了上述三个项，再加上价值函数的MSE损失，PPO目标定义如下：


$$
\mathcal{L}_{\text{PPO}}(\theta, \gamma) = \underbrace{\mathcal{L}_{\text{clip}}(\theta)}_{\text{最大化奖励}} + \underbrace{w_1 H(\theta)}_{\text{最大化熵}} - \underbrace{w_2 \text{KL}(\theta
)}_{\text{惩罚KL散度}} - \underbrace{w_3 \mathcal{L(}\gamma)}_{\text{批评者L2}}
$$

以下是该目标中不同项的总结：


| **术语**               | **目标**                                                             
|-------------------------|-----------------------------------------------------------------------------|
| $\mathcal{L}_{\text{clip}}(\theta)$     | 最大化高优势动作的奖励（剪裁以避免不稳定）。 |
| $H(\theta)$       | 最大化熵以鼓励探索。                                  |
| $\text{KL}(\theta)$           | 惩罚与参考策略的偏差（稳定性）。                  |
| $\mathcal{L}(\gamma)$    | 最小化价值预测的误差（批评者L2损失）。

---

# RL算法：GRPO

现在我们对 PPO 有了很好的理解，理解 GRPO 就非常简单了

关键的区别在于两种算法如何估计优势 $A$：与 PPO 通过价值函数估计优势不同，GRPO 通过从LLM中使用相同的prompt进行多次采样来实现这一点。

<v-click>

**工作流程：**
1. 对于每个prompt $p$，从LLM策略 $\pi_\theta$ 中采样一组 $N$ 个响应 $\mathcal{G}=\{r_1, r_2,...r_N\}$；
2. 使用奖励模型 $R_\phi$ 计算每个响应的奖励 $\{R_\phi(r_1),R_\phi(r_2),...R_\phi(r_N)\}$；
3. 计算每个响应的组内归一化优势：
$$
\begin{align}
A_i = \frac{R_\phi(r_i) - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})},
\end{align}
$$
其中 $\text{mean}(\mathcal{G})$ 和 $\text{std}(\mathcal{G})$ 分别表示组内的均值和标准差。

</v-click>

---

# GRPO 目标函数
与PPO类似

GRPO仍然使用了**剪裁的代理损失**以及**KL惩罚**。
这里没有使用熵奖励项，因为基于组的采样已经鼓励了探索。剪裁的代理损失与PPO中使用的相同，但为了完整性，这里再次列出：
$$
\begin{align*}
& \mathcal{L}_{\text{clip}}(\theta) =  \\ 
&\frac{1}{N} \sum_{i=1}^N \left( \min\left( \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)} A_i, \ \text{clip}\left( \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)}, 1-\epsilon, 1+\epsilon \right) A_i \right) \right),
\end{align*}
$$


<v-click>

然后加上KL惩罚项，最终的GRPO目标函数可以写成：

$$
\begin{align}
\mathcal{L}_{\text{GRPO}}(\theta) &= \underbrace{\mathcal{L}_{\text{clip}}(\theta)}_{\text{最大化奖励}} - \underbrace{w_1\mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{orig}})}_{\text{惩罚KL散度}}
\end{align}
$$

</v-click>

<v-click>

**是不是简单多了？** 在GRPO中，优势近似为每个响应在其组内响应中的归一化奖励。这消除了批评网络计算每步奖励的需要，更不用说数学上的简单和优雅了。这不禁让人产生疑问——为什么我们没有早点这么做？

</v-click>

---
layout: section
---

# 关于R1的更多思考：残酷的简单

---

# 关于R1的更多思考：残酷的简单
关于R1的工程多说几句。

无论是否被过度炒作，阅读论文时R1的一个真正突出的特点是它采用了一种**剥离繁琐、不加修饰的方法**来进行LLM训练，优先考虑简单性而非复杂性。GRPO只是冰山一角。这里还有一些其他体现其残酷简单性的例子：

<v-click>

## **1. 基于规则的确定性奖励**
- **What**：放弃神经过程奖励模型（PRMs）或结果奖励模型（ORMs）。使用**二进制检查**，包括：
  - **答案正确性**：最终答案与真实答案匹配（例如，数学解法，代码编译）。
  - **格式化**：强制将答案格式化为 `<think>...</think><answer>...</answer>` 模板。
  - **语言一致性**：惩罚混合语言输出（例如，为中文查询提供英文推理）。
- **Why**：确定性规则避免了**奖励作弊**（例如，模型通过看似合理但错误的步骤欺骗神经奖励模型），并消除了奖励模型的训练成本。

</v-click>

---

# 关于R1的更多思考：残酷的简单

<v-click>

## **2. 冷启动数据：最小化人工干预**
- **What**：与其策划大规模SFT数据集，不如通过以下方式收集**几千个高质量的CoT示例**：
  - 通过少样本示例提示基础模型。
  - 轻量级人工后处理（例如，添加markdown格式）。
- **Why**：避免了昂贵的SFT阶段，同时通过“足够好”的起点引导RL。

</v-click>

<v-click>

## **3. 拒绝采样：过滤困难，训练更难**
- **What**：在RL训练后，生成**60万个推理轨迹**，然后**丢弃所有错误响应**。只保留“获胜者”（正确答案）用于监督微调（SFT）。没有复杂的重新排序，没有偏好对。只是适者生存的过滤。
- **Why**：它有效，为什么不这样做呢！

</v-click>

---

<v-clicks>

DeepSeek-R1 的设计反映了 AI 领域的一个更广泛趋势：**规模和简单性往往胜过复杂的工程**。

通过毫不留情地简化——用规则替换学习组件，利用大规模并行采样，并锚定在预训练基线上——R1 以更少的失败模式实现了 SOTA 结果。

它并不优雅，但它是**有效的**。

谁会想到激励良好思考的最佳方式是：**停止过度思考**。

</v-clicks>

---
layout: end
---

# 请各位老师批评指正！