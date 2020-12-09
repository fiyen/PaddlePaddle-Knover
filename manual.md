# 这是一个比较详尽的Knover使用手册
该项目是从项目集“没有对象就自己造”提炼总结出来的，感兴趣的请参考[链接](https://aistudio.baidu.com/aistudio/projectdetail/542430)的“没有对象就自己造板块”。这个集合主要的目的是记录我关于Knover的使用心得，并为以后使用这个工具的人提供一些快速入门的建议。其中难免有一些错误的地方，还请见谅。
# 1 什么是Knover
Knover是基于飞桨的开放域对话模型训练工具箱。通过Knover，我们可以轻松训练出一个开放域对话模型，并支持多种输入格式，支持单文件和多文件读取。同时，Knover提供了基于飞桨的优秀的优化训练策略，加速模型的训练。目前，Knover已经支持PLATO-2模型的训练。
Knover官网：https://github.com/PaddlePaddle/Knover 。
# 2 什么是PLATO-2
Plato是百度推出的一个基于飞桨的多轮对话模型。该模型的最大改进在于通过引入离散隐变量实现了对话场景中多回答中的择优，即，对于同一个问题，实现不同场景下的不同回答的选择。最新推出的Plato-2在中英文效果上，已全面超越 Google Meena、Facebook Blender、微软小冰等先进模型。
模型的整体框架如图所示。该模型采用了经典的Transformer结构，通过注意力机制提高了模型针对不同长度对话的生成效果。隐变量z的引入，使预训练模型依据z可以生成多种回答，最终回答从多种回答中择优。在训练中，该模型采用两阶段训练的方法。第一阶段，基于表现良好的预训练模型，训练出一对一回答的模型；第二阶段，引入评估和隐变量z，训练出一对多回答的模型。模型的具体原理可以参考原论文，论文地址：https://arxiv.org/abs/2006.16779
。
![Plato-2](https://github.com/fiyen/SomePictures/blob/main/Plato-2ModelReview.png)
# 认识Knover
