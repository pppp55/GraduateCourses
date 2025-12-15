# RL for RS: Literature Review

### 1 Background

The recommendation system, as a key bridge connecting users and content, has evolved from a simple "recommending a product" to a complex decision-making process of "serializing and recommending a series of products". With the explosive growth of the number of products on e-commerce platforms and the abundance of consumer behavior data, traditional recommendation methods (such as DIN and MMOE) can only provide instant recommendations of "what is good immediately", but cannot solve the long-term value problem of "what is good in the long run". Reinforcement learning (RL) has become an effective tool for solving long-term optimization problems in recommendation systems due to its ability to model sequential decision-making processes.

推荐系统作为连接用户与内容的关键桥梁，已从简单的"推荐一个商品"演变为"序列化推荐一系列商品"的复杂决策过程。随着电子商务平台商品数量爆发式增长和消费者行为数据的丰富，传统推荐方法（如DIN、MMOE）仅能提供"what is good immediately"的即时推荐，却无法解决"what is good in the long run"的长期价值问题。强化学习（RL）因其能建模序列决策过程的特性，成为解决推荐系统长期优化问题的有效工具。

### 2 核心研究脉络

##### 2.1 The Evolution of Reinforcement Learning in Recommendation Systems

Based on the review of "Reinforcement Learning based Recommender Systems: A Survey", the application of RL in recommendation systems has gone through three main stages:

Basic exploration period (2015-2016): Preliminary attempts based on DQN and DDPG (such as Slate MDP, Wolpertinger)

Method optimization period (2017-2018): Introducing more complex DRL algorithms and state representations (such as CEI, SRL-RNN)

Application Extension Period (2019-2021): Algorithms applied to specific fields such as healthcare, fair recommendation, knowledge graphs, etc. (such as MASSA, FairRec, KGRL)

基于《Reinforcement Learning based Recommender Systems: A Survey》的综述，RL在推荐系统中的应用经历了三个主要阶段：

基础探索期（2015-2016）：基于DQN和DDPG的初步尝试（如Slate-MDP, Wolpertinger）

方法优化期（2017-2018）：引入更复杂的DRL算法和状态表示（如CEI, SRL-RNN）

应用扩展期（2019-2021）：算法应用于医疗健康、公平推荐、知识图谱等特定领域（如MASSA, FairRec, KGRL）

##### 2.2 Evolution of State Representation Methods

The paper points out that state representation is the core of RLRS, which has mainly evolved from SR1 (raw features) to SR2 (user item interaction matrix) and then to SR3 (dense, low dimensional vector representation). SR3 has become the dominant solution for DRL based RSs, enabling deep learning models to be trained more effectively.

论文指出，状态表示是RLRS的核心，主要经历了从SR1（原始特征）到SR2（用户-物品交互矩阵）再到SR3（密集、低维向量表示）的演进。SR3成为DRL-based RSs的主导方案，使得深度学习模型能够更有效地进行训练。

##### 2.3 Optimization of Exploration Strategies

From early ε - growth to later entropy regularization exploration strategies (such as MASSA), exploration strategies have been continuously optimized to better balance exploration and utilization. The book "Deep Exploration for Recommendation" further emphasizes the importance of "deep exploration" and points out that traditional "single step exploration" cannot effectively capture the long-term value of recommendation systems.

从早期的ε-greedy到后来的熵正则化探索策略（如MASSA），探索策略不断优化以更好地平衡探索与利用。《Deep Exploration for Recommendation》进一步强调了"深度探索"的重要性，指出传统"单步探索"无法有效捕捉推荐系统的长期价值。

### 3 代表性工作与创新点

##### 3.1 RecoMind: LLM Empowered Recommendation Agents

《RecoMind: Large Language Model Powered Agent For Recommendation》 A recommendation agent based on a large language model has been proposed, and its core innovation lies in:

Self Inspiring (SI) planning method: preserves all previous states in the historical path when generating new states, unlike CoT and ToT

Three module architecture: Planning, Memory and Tools, making full use of LLM's reasoning ability and tool calling ability

Dynamic tool invocation: integrates database tools, search tools, and text summarization tools to expand the knowledge boundaries of recommendation systems

《RecoMind: Large Language Model Powered Agent For Recommendation》提出了一种基于大语言模型的推荐智能体，其核心创新在于：

Self-Inspiring (SI)规划方法：在生成新状态时保留所有历史路径中的先前状态，与CoT和ToT不同

三模块架构：Planning、Memory和Tools，充分利用LLM的推理能力和工具调用能力

动态工具调用：整合数据库工具、搜索工具和文本总结工具，扩展了推荐系统的知识边界

##### 3.2 Deep Exploration: Long term value oriented recommendations

The Deep Exploration for Recommendation emphasizes that recommendation systems should be modeled from the perspective of "sequential decision problems" rather than "single step decision problems". Its core contributions include:

Deep exploration concept: Consider the impact of recommendation sequences on long-term user satisfaction, rather than just focusing on immediate feedback

High fidelity industrial grade simulation: validated the effectiveness of deep exploration through industrial grade simulators

Long term value optimization: Experimental evidence shows that deep exploration methods can significantly improve the long-term value of recommendation systems, not just short-term click through rates

《Deep Exploration for Recommendation》强调推荐系统应从"顺序决策问题"角度建模，而非"单步决策问题"

其核心贡献包括：

深度探索概念：考虑推荐序列对用户长期满意度的影响，而非仅关注即时反馈

高保真工业级模拟：通过工业级模拟器验证了深度探索的有效性

长期价值优化：实验证明深度探索方法能显著提高推荐系统的长期价值，而不仅仅是短期点击率

### 4 研究趋势与融合

##### 4.1 Integration of Reinforcement Learning and Large Language Models

RecoMind represents a new trend in the integration of RL and LLM, combining the reasoning and tool calling capabilities of LLM with the decision optimization capabilities of reinforcement learning, providing a new development path for recommendation systems.

《RecoMind》代表了RL与LLM融合的新趋势，将LLM的推理能力和工具调用能力与强化学习的决策优化能力结合，为推荐系统提供了新的发展路径。

##### 4.2 From short-term indicators to long-term value

Deep Exploration reveals a paradigm shift in recommendation system research from focusing on short-term metrics such as click through rates to focusing on long-term values such as user satisfaction and retention rates. This is an important paradigm shift in recommendation system research.

《Deep Exploration》揭示了推荐系统研究从关注短期指标（如点击率）向关注长期价值（如用户满意度、留存率）的转变，这是推荐系统研究的重要范式转变。

##### 4.3 From single model to systematic architecture

The research has developed from the initial single point algorithm improvement (such as DQN) to the systematic architecture design (such as the three module architecture of RecoMind), emphasizing the optimization of the recommendation system as a whole.

研究从最初的单点算法改进（如DQN）发展到系统化架构设计（如RecoMind的三模块架构），强调了推荐系统作为整体的优化。

### 5 研究挑战与未来方向

##### 5.1 Current Challenges

Reward function design: How to design a reward function that reflects both immediate user feedback and captures long-term value

Exploration efficiency: Efficiently explore recommendation space within a limited number of interactions

System Scalability: How to Design Scalable Distributed Architecture to Support Large Scale Recommendation Problems

奖励函数设计：如何设计既能反映用户即时反馈又能捕捉长期价值的奖励函数

探索效率：在有限交互次数内高效探索推荐空间

系统可扩展性：如何设计可扩展的分布式架构以支持大规模推荐问题

##### 5.2 Future direction

Multi objective optimization: simultaneously optimizing multiple business objectives (such as click through rate, retention rate, diversity)

Fairness and interpretability: Introducing fairness and interpretability considerations in reinforcement learning recommendations

Cross disciplinary migration: Transfer RLRS technology to more fields (such as healthcare, education)

Deep Fusion of LLM and RL: Exploring the Deeper Application of LLM in Recommendation Systems

多目标优化：同时优化多个业务目标（如点击率、留存率、多样性）

公平性与可解释性：在强化学习推荐中引入公平性和可解释性考量

跨领域迁移：将RLRS技术迁移到更多领域（如医疗健康、教育）

LLM与RL的深度融合：探索LLM在推荐系统中更深层次的应用

### 6 总结

The research on recommendation systems based on reinforcement learning has evolved from early algorithm exploration to systematic architecture design, with a focus on "how to use RL" to "how to use RL to solve the core problems of recommendation systems". Currently, research is undergoing three important transformations: from short-term indicators to long-term value, from a single algorithm to a systematic architecture, and from traditional RL to integration with LLM. These transformations not only drive the development of recommendation system theory, but also provide more effective recommendation solutions for the industry.

基于强化学习的推荐系统研究已从早期的算法探索发展为系统化的架构设计，研究重点从"如何用RL"转向"如何用RL解决推荐系统的核心问题"。当前研究正经历三个重要转变：从短期指标到长期价值、从单一算法到系统化架构、从传统RL到与LLM的融合。这些转变不仅推动了推荐系统理论的发展，也为工业界提供了更有效的推荐解决方案。

### 7 Slides Structure

Title Page: Research on Reinforcement Learning Based Recommendation Systems: Literature Review

Research Background and Significance (1 page)

Core research framework (2 pages, showcasing the development stages of RLRS)

Representative works and innovative points (2 pages, with a focus on introducing three papers)

Research Trends and Integration (1 page)

Research Challenges and Future Directions (1 page)

Summary (1 page)

标题页：基于强化学习的推荐系统研究：文献综述

研究背景与意义（1页）

核心研究脉络（2页，展示RLRS的发展阶段）

代表性工作与创新点（2页，重点介绍三篇论文）

研究趋势与融合（1页）

研究挑战与未来方向（1页）

总结（1页）