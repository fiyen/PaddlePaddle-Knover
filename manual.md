# 这是一个比较详尽的Knover使用手册
该项目是从项目集“没有对象就自己造”提炼总结出来的，感兴趣的请参考[链接](https://aistudio.baidu.com/aistudio/projectdetail/542430)的“没有对象就自己造板块”。这个集合主要的目的是记录我关于Knover的使用心得，并为以后使用这个工具的人提供一些快速入门的建议。其中难免有一些错误的地方，还请见谅。
# 1 什么是Knover
Knover是基于飞桨的开放域对话模型训练工具箱。通过Knover，我们可以轻松训练出一个开放域对话模型，并支持多种输入格式，支持单文件和多文件读取。同时，Knover提供了基于飞桨的优秀的优化训练策略，加速模型的训练。目前，Knover已经支持PLATO-2模型的训练。
Knover官网：https://github.com/PaddlePaddle/Knover 。
# 2 什么是PLATO-2
Plato是百度推出的一个基于飞桨的多轮对话模型。该模型的最大改进在于通过引入离散隐变量实现了对话场景中多回答中的择优，即，对于同一个问题，实现不同场景下的不同回答的选择。最新推出的Plato-2在中英文效果上，已全面超越 Google Meena、Facebook Blender、微软小冰等先进模型。
模型的整体框架如图所示。该模型采用了经典的Transformer结构，通过注意力机制提高了模型针对不同长度对话的生成效果。隐变量z的引入，使预训练模型依据z可以生成多种回答，最终回答从多种回答中择优。在训练中，该模型采用两阶段训练的方法。第一阶段，基于表现良好的预训练模型，训练出一对一回答的模型；第二阶段，引入评估和隐变量z，训练出一对多回答的模型。模型的具体原理可以参考原论文，论文地址：https://arxiv.org/abs/2006.16779 。
![Plato-2](https://github.com/fiyen/PaddlePaddle-Knover/blob/main/pictures/Plato-2ModelReview.png)
# 3 认识Knover
## 3.1 认识Knover的主要文件
对于Knover的使用，用命令行是比较方便的（当然如果有能力可以调用sh文件）。根目录的三个文件：train.py,test.py和save_inference_model.py是最经常用到的。从名称也可以看出来，train.py用来训练，test.py用来测试，save_inference_model用来导出预测用的模型，这一个是在模型训练完以后，部署训练好的模型，又希望保持模型尽可能缩减不必要的参数时用的。

1.package文件夹中存放了其自带的试验数据的词集，语句切分模型（spm.model, 即sentencepiece model，这个模型用在语句的预处理上，必须给定），以及模型的细节参数（词集大小，隐含层个数，激活函数等等，具体查看package/dialog_en/24L.json。
2.models文件夹存放了模型的各个子模块，plato模块也在其中
3.data文件夹存放了实验用的小文件
4.tasks文件夹中包含了模型两种应用任务的实现，包括“下一句语句预测”和“会话生成”。这个应用任务的选择是必须给出的，对应参数 --tasks, 分别写作NextSentencePrediction和DialogGeneration。具体来说，DialogGeneration是训练对话模型时用到的，而NextSentencePrediction是在训练打分模型时用到的。这里的具体区别后边再讲。

## 3.2 认识Konver的主要参数
`--init_pretraining_params` 预训练模型所在文件夹，如果需要加载（当然需要）预训练模型需要给出这个参数

`--init_checkpoint` 保存节点的所在文件夹，如果给出了init_checkpoint则从该文件夹初始化训练参数（等于覆盖了init_pretraining_params的参数），checkpoint保存了模型更多的细节，如训练步数，当前学习率，还有模型涉及的所有训练参数等，如果从checkpoint开始继续训练模型，模型会从之前中断的状态继续训练，如果不设--start_step模型会错误显示当前的步数，但是内部的参数是按照正确的步数更新的。

train.py

`--train_file` 训练文件地址

`--valid_file` 评估文件地址

`--model` 用到的模型名称：`Plato`：plato；NSPModel：next_sentence_prediction model

`--config_path` 模型细节参数配置文件，如24L.json

`--task` 模型应用任务 NextSentencePrediction；DialogGeneration；UnifiedTransformer

`--vocab_path` 词集路径

`--spm_model_file` sentencepiece model文件的路径

`--num_epochs` 训练周期数

`--log_steps` 输出训练详情的间隔步数

`--validation_steps` 评价间隔步数

`--save_steps` 保存间隔步数

infer.py

`--infer_file` 需要推断的文件

`--output_name` 需要保存的对象，response；data_id；score

`--model` 用到的模型名称：Plato：plato；NSPModel：next_sentence_prediction model

`--config_path` 模型细节参数配置文件，如24L.json

`--task` 模型应用任务 NextSentencePrediction；DialogGeneration；UnifiedTransformer

`--vocab_path` 词集路径

--spm_model_file sentencepiece model文件的路径

## 3.3 了解对话模型的训练
### 3.3.1 一般模型的训练
第一步：在进行训练之前，需要提前准备好几个文件：详列模型参数的.json文件，如config文件夹下的24L.json文件；分词工具的预训练模型，如spm.model文件；以及分词后形成的词表文件，如vocab.txt。

第二步：准备数据。把自己准备的训练数据转换成合适的格式，存入txt文件，供训练使用。

第三步：调用train.py进行训练，模型选择UnifiedTransformer，任务选择DialogGeneration。

第四步：训练完成后，调用save_inference_model.py将预测模型导出

经过以上四步，模型就训练好了。当然这个过程需要巨量的训练集做支撑才能训练出好的模型。
### 3.3.2 Plato-2模型的训练
在上述四步完成后进行。前两步与上述过程相似，在训练出UnifiedTransformer模型后，按照以下步骤进行训练：

第三步：调用train.py进行训练，模型选择Plato，任务选择DialogGeneration，并且--init_pretraining_params选择之前训练好的UnfiedTransformer模型（如果未进行上述第四步，则导出的模型可以用--init_checkpoint指定）

第四步：训练完成后，继续调用train.py进行训练，模型选择Plato，任务选择NextSentencePrediction（注意区别）。训练打分模型。

第五步：分别用save_inference_model.py导出PLATO模型和打分模型NSP。

经过这些步，模型训练完成。用infer.py预测时，如果使用PLATO模型，需要指定打分方使--ranking_score，如果选择nsp_score，则需要设定打分模型为NSP。以上具体过程后边会细讲。

# 4 具体操作
## 4.1 数据准备
Plato-2模型的输入采用了token，role，turn，position相融合的表示方式。在训练和测试过程中，我们需要搞清楚文本数据需要经过怎样的转化才能作为输入，以及输出数据需要怎样的处理才能转换成文本。目前我们可以获取各种开放的对话训练集，如微博，腾讯，华为等提供的比赛用的数据集。
## 4.1.1 中文分词
中文必须面对的一个问题就是如何实现分词。在公开的开放域对话数据集中，大多数已经做了分词，然而真实场景中语句是不可能时时刻刻都被分词了的。在Knover的源码中，对输入的处理是通过了sentencepiece工具（BERT也使用的这个）。sentencepiece提供了一种方便快捷的分词操作，我们可以直接将整个数据集放进去，设定分词的单元数量，然后等待训练出一个好用的分词模型（会输出一个预训练模型，之后对每个语句都会用这个模型进行编码和解码，即分词，转换成数字编码，输出再转换回句子）。Knover中训练必须输入的参数spm_model，就是对应sentencepiece的预训练模型。我们当然可以自己训练一个sentencepiece的预训练模型出来，但是考虑到分词模型对效果的影响，推荐大家使用千言多技能对话中给出的baseline模型（luge-dialogue）中附带的spm.model文件，这个文件分词的效果已经非常出色了。当然，别忘了搭配词表vocab.txt使用。目前这个比赛已经关闭，luge-dialogue这个模块可以在Konver官网获得。

仔细分析luge的spm.model我们可以发现，这个预训练模型其实是根据已经分词的句子训练的，虽说如此，因为分词单元足够多，也覆盖了所有常见的单个中文词。我们可以直接把语句送入得到编码，也可以先用jieba分词预先分一次（也可以用其他分词工具），然后再编码。用sentencepiece模型的例子如下（文件exams/do_sentencepiece.py）：

```
import sentencepiece as sp
import jieba
text = "我今天做了一顿丰盛的晚餐！"

SPM = sp.SentencePieceProcessor()
SPM.load('spm.model')
# 直接分词
ids = SPM.encode_as_ids(text)
print(ids)
print(SPM.decode_ids(ids))

# 先用jieba分词，再用sentencepiece编码
text = ' '.join(list(jieba.cut(text)))
ids = SPM.encode_as_ids(text)
print(ids)
print(SPM.decode_ids(ids))
```
