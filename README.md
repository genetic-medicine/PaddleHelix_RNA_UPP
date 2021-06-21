# 螺旋桨RNA结构预测竞赛第一名方案

## 竞赛要求
基于百度发布的RNA二级结构预测算法LinearFold和RNA配分方程算法LinearPartition，预测给定RNA序列在每个位点上保持不成对的概率.

详细请见： https://aistudio.baidu.com/aistudio/competition/detail/61

aistudio源代码公开: https://aistudio.baidu.com/aistudio/projectdetail/1479469

## 生物基础 ## 
*注: 如有生物背景,请跳过本节*

所有可以称为生命体的系统，无论简单或复杂，无一例外地以核酸为遗传物质。核酸主要分为脱氧核糖核酸(DNA)和核糖核酸(RNA)两大类，每类核酸都是由四种不同的单元（称为碱基）以线性的形式连接成一个长链，它们的排列顺序就是我们所说的基因序列。核酸在自然界中的重要地位毋庸置疑，甚至于无自我繁殖能力的病毒也是以核酸作为基因，比如正在全世界肆虐的新冠病毒的基因是大约3万个碱基长的单链RNA. 有些基因能够编码特异的蛋白，我们称这些为编码基因. 分子生物学的中心法则描述了从基因到蛋白的过程: DNA先被转录为信使RNA (mRNA), 然后mRNA被翻译为蛋白。但编码基因只占了基因组的一小部分(~2%)，大部分被转录的RNA是非编码RNA。

RNA是本次螺旋浆结构预测竞赛的主角。竞赛要求用深度学习的方法预测每个碱基的不配对几率。 作为一个线性的链状大分子，RNA具有非常高的柔性， 易于弯曲， 好比一条细长的线。 如果碱基之间没有任何相互作用，RNA在三维空间里就会杂乱无章毫无结构而言。可作为生物大分子的RNA需要具有特定结构才能行使其生物功能, 因而碱基间是否配对和如何配对至关重要.

RNA主要由四种碱基组成，虽然我们最熟悉的是两种碱基对类型, 真实的情况是碱基有很多配对的方式. 每个碱基有三个边可以配对，再加上糖苷键的空间取向，两个碱基间就有12种不同的配对方式, 全部一起有超过30种不同的碱基两两配对组合(这里不考虑概率极小的两个以上碱基的配对)。所以对于一个RNA序列, 其可能的配对方式是一个巨大的排列组合空间。 碱基之间配对的源驱动力来自于碱基之间的吸引力(直接或间接)，最主要的方式是氢键。每一个碱基和其它碱基都存在或强或弱的吸引力，但是只能和其中一个配对，所以它们相互竞争, 最优的配对组合能最大化碱基之间的吸引作用, 从热力学角度上说自由能最小化. 同时往往有很多配对组合有相近的能量, 并不存在一个单一的稳定结构, 而是有很多亚稳定结构. 这时候一个非常有意义的碱基特性就是它在这些结构集里的不配对几率(或者配对几率).

## 赛题说明 ##
### 技术基础 ###
百度螺旋桨团队近年来在RNA结构预测上做出了重要的贡献, 研发出速度最快同时准确度极高的一系列线性算法, 做到了对RNA二级结构的直接预测和比对(LinearFold[1]), 对碱基配分和配对几率的预测(LinearPartition[2]), 和对RNA序列的设计优化(LinearDesign[3]).  这次比赛主要是基于螺旋桨团队的LinearFold和LinearPartition算法，用深度学习的方法来预测RNA的不配对几率. 

在LinearDesign文章里对碱基不配对几率的意义有很好的详述, 拿mRNA疫苗为例, 如果mRNA序列里大部分碱基有很小的不配对几率, 那说明这是一个稳定的二维甚至三维RNA结构. 稳定的结构会大大延长其在细胞内的寿命,从而提高蛋白的产量. 如果大部分碱基有非常大的不配对几率, 那说明这是一个非常无结构或动态的RNA. 这会让mRNA在溶液中易于被水解, 降低其稳定性和寿命, 不利于运输,储藏,翻译等等. 所以碱基不配对几率对RNA序列设计有极其重要的定量指导意义.

### 训练数据分析 ###

训练数据集一共4750个RNA序列， 验证集250个序列, 测试集112个序列，它们的长度分布如图一所示. 训练和验证集长度在100 和500碱基之间, 分布相对均匀，可是测试集里有较长的序列，尤其超过30%的序列的长度500以上，这给预测增加了难度。如果按照碱基计算，训练和验证集一共有1,562,736个碱基(序列平均长度313碱基)，测试集一共有49,279碱基（序列平均长度440碱基）。 

![图一](https://ai-studio-static-online.cdn.bcebos.com/1785b0ba9f464b85a36211a9d8cc44e2012a563dbff641aabb6f4c683f74314a)

图表 1: 训练(蓝),验证(红)和测试(绿)集RNA序列的长度分布.

基于提供的序列, 我们用LinearFold计算二级结构和用LinearPartition计算不配对几率。这些计算的结果可以转换成碱基不配对几率, 比如LinearFold的二级结构可转换为二进制的不配对几率(0或1). LinearFold对训练，验证和测试集预测的碱基不配对几率分布大体一致，约62%的碱基是配对的，其余38%不配对. LinearPartition所预测的不配对几率是一个连续空间，其分布如图二所示。我们可以看到三个数据集的几率分布相差不多，最突出的特征是几率分布非常不均匀, 尤其有两个分布峰分别在0和1附近, 说明即使在连续空间里大多数碱基的配对几率集中在0或1附近. 为了能直接比较LinearFold和LinearPartition, 我们可以把LinearPartition的连续空间二进制化 (0.0-0.5为0, 0.5-1.0为1），这样得到～55%的碱基不配对几率为0， 和LinearFold预测的62%有一些差距. 

![](https://ai-studio-static-online.cdn.bcebos.com/c8d6c171d7a44f2e9bca418c22f336d323df11c6c4644cd7944ee98f793a67dd)

图表 2: LinearPartition预测的碱基不配对几率分布: 训练(蓝), 验证(红)和测试(绿)集.

训练和验证集里的标注数据是每个碱基的不配对几率。图三展示了聚集了所有序列碱基不配对几率的分布, 我们可以看到不配对几率成两极分化走势， 绝大部分碱基的配对或不配对的几率在95%以上，这比LinearPartition预测的不配对概率分布要更加两极分化。值得一提的是标注数据在二进制离散化后的分布和LinearFold预测高度一致，给出～60%碱基配对.

![](https://ai-studio-static-online.cdn.bcebos.com/b49c78dfc040408da4f09cad92a37b0cfd102db22a854cf1b2f81d9149528206)


图表 3: 训练(蓝)和验证(红)集碱基不配对几率标注分布

### 评比指标 ###

碱基不配对几率预测和标注的差异采用的是经典而直观的均平方差的平方根（Root Mean Square Error, RMSE)。如前文所述， 我们可以把LinearFold和LinearPartition的预测结果转换为碱基的不配对几率， 这样可以先和标注作对比。 虽然LinearFold给出结果的整体分布和标注几乎形同，其RMSE却有～0.30， 而LinearPartition给出的RMSE稍小，～0.26. 可见在预测连续空间的不配对几率时， LinearPartition有更好的表现， 虽然其分布目测差距更大。 当然我们还可以继续分析RMSE和RNA序列长度的相关性等等， 这里不做详述。

## 模型构建和训练 ##

### 框架思路 ###
这个深度学习任务是一个典型的序列到序列(seq2seq or seq2vec)问题： 输入是RNA的碱基序列， 输出是每个碱基的不配对概率。那么，一个序列到序列的深度学习模型需要“学到”碱基之间什么样的关系才能准确预测RNA的不配对几率呢？我们从以下几个角度考虑碱基配对问题, 并设计了对应的网络框架。

第一，一个碱基的配对与否是由全部序列整体决定的，仅仅利用序列的任何片段的信息是不能准确预测的。这个特性可以用循环神经网络（RNN）或者变压器（Transformer）来描述。我们选择了一个Transformer编码器层来赋予每个碱基一个全局的信息，对位置的编码用的是正旋和余旋函数。
第二，碱基的排列是一个有极强线性的序列，改变碱基间的任何顺序都会或大或小地影响其不配对概率，即使有些碱基变化能保持碱基的二级甚至三极的最稳定结构，碱基的不配对几率也不会丝毫不变。所以我们决定加入一个RNN来强化碱基间的序列依赖， 具体形式采用了一个双向LSTM层。 另外一个考虑是Transformer编码器在训练数据有限情况下可能会表现不佳。
第三，碱基配对有较强的局部相关， 要形成稳定的碱基配对， 至少相邻的三个以上的碱基都配对， 自然越多的连续碱基配对越稳定，同时只有1-2个连续的碱基有很小的不配对几率是不可能的。 鉴于此， 我们采用了一维的卷积层(Conv1D)来描述这种局部相关性。 

基于以上设计理念，如图四所示, 除了输入嵌入层外, 模型框架有以下组成。1） 输入全连接模块，2）Transformer编码模块， 3）双向LSTM模块， 4） Conv1D模块， 5）输出全连接模块. 对模型的深度和宽度的考量有两个主要方面。 一是训练和验证集的数量有限，标注的碱基个数一共有～1.6M，过大的模型容易过拟合， 二是LinearFold/LinearPartition的预测给了很好的起始点， 对模型优化有较好的引导。我们最后采用的参数为, 模块1-4的层数为1, 维度为32 ，输出全连接模块层数为3, 维度分别为32, 32, 2, 归一化为LayerNorm，激活函数为ReLU, Dropout为0.2, 最终可训练的参数个数为36,418, 大约是标注个数的2.2%。

<img src="https://ai-studio-static-online.cdn.bcebos.com/7227509d29654c22ad88326a918926acf836ab482a3045f58bb6fdc64d1245c3" width=480/>

图表 4: 模型框架示意图

### 模型实现 ###

我们用飞浆2.0深度学习平台(Paddle)实现本模型， 过程中大量参考了飞浆的详细文档和开源的源代码， 尤其利用了Paddle.nn中直接可用的TransformerEncoder, LSTM, Conv1D等等。损失函数也主要采用Paddle.nn提供的函数库。为了更便捷的搭建不同的模型或改变模型的参数，我们把主要的参数存储在一个args结构里，然后包装每个模块（比如LSTM, Conv1D)能接受args为输入， 图5展示了竞赛所用模型的代码，图6展示了其中一个模块的代码, 搭建过程相对简单, paddle_nets.py包含了所有模型建构的相关代码.

![](https://ai-studio-static-online.cdn.bcebos.com/851f3bb5c1774cedbddfa4a6e5dd9aec3c091f79f06b451d93af31ed7c66d634)

图表 5: 网络框架构建代码范例

<img src="https://ai-studio-static-online.cdn.bcebos.com/a83239417f1b4bfa983598977f2385952e20f162246d4e87b1a406501ed4d46b" width=800/>

图表 6: 模块构建代码范例

### 输入输出 ###
对于一个长度为L的RNA序列 每个碱基字符用一个四维的onehot向量表达， LinearFold的二级结构字符用一个四维的onehot向量表达(三维足够,这里凑了个偶数)，LinearPartition预测可以直接使用。最终得到的输入矩阵的维度为 [N, L, 10], 其中N是batch size, L是序列长度。 输出全连接模块的最后一层维度分别为2. 所以模型的输出是一个[N, L, 2]维的矩阵， 沿最后一维进行softmax操作，得到了每个碱基的配对和不配对几率。损失函数有两种可取的方法，最直接的是对不配对几率计算竞赛采用的均平方差的平方根（RMSE）， 同时考虑结果为softmax后的几率，标注为二分类的软标注，我们也可以采用交叉熵做为损失。 两种不同主要在于交叉熵在对几率p的梯度上多了一个1/p(1-p)的因子，对接近0或1的几率梯度增强。 在实验中我们发现两种损失函数的结果是大体相同， 猜想是因为几率主要分布在0和1的附近， 这个附加因子的作用不显著. 

### 模型训练 ###
模型训练在单个CPU上进行, 每个epoch大约5分钟，一般一两个小时能训练结束. 有关代码在fly_paddle.py脚本里. 前文提到的args包含了大部分模型和训练的参数, fly_paddle.py设置了缺省参数,  同时可以在命令行设置参数取值, 图7演示了命令行启动模型训练的界面, 训练时汇报损失界面如图8所示.

![](https://ai-studio-static-online.cdn.bcebos.com/8ff0490f54d740148e120f2577e9c39afb3ea482d8da4c82b809ef55f7b313fc)

图表 7: 命令行启动模型训练范例

![](https://ai-studio-static-online.cdn.bcebos.com/0c176a62798c4b80ba4ff86926f297f6ef045abb0a684e48bc0926dca4967f51)

图表 8: 模型训练中的进度汇报范例

模型的训练主要分以下三步骤:
第一步是对深度学习任务培养一些直觉。我们先用最简单的全连接模型看一下在不考虑碱基间相互作用情况下的RMSE。如果输入的碱基信息只有其字符编码的四维的onehot向量， RMSE为～0.42，不出所料和随机猜测几率相差无几，加入LinearFold和LinearPartition预测信息能降低RMSE到0.25左右。可见LinearFold和LinearPartition给出了很好的初始点，如果要进一步降低RMSE, 我们需要考虑碱基之间的相互作用.

第二步是分步加入TransformerEncoder, LSTM和Conv1D模. 我们发现单一的TransformerEncoder或者LSTM足够降低RMSE到0.21-0.22之间，Conv1D模块对RMSE影响较小, 这与我们的期望也大致相符，因为一层的Conv1D也只是考虑到了紧邻碱基之间的作用。最后三个模块一起能降低RMSE到0.20-0.21区间。

第三步主要是摸索超参数和改进模型来进一步降低RMSE。我们首先对learning_rate， dropout, 和weight decay做了一系列的摸索， 发现在常用的超参数区间里模型表现相当， 原因可能主要有两个，一个是参数空间相对较小，其二是起始的数据和标注相距不远. 因为训练集和验证集的RMSE在所有的训练中RMSE相当, 我们没有采用参数正则化.  

我们对改进模型的努力收获较小。图9展示了对训练集序列中模型预测的碱基不配对几率分布， 最突出的问题是有39%的碱基的几率在[0.1, 0.9]区间内，而标注集中只有22%碱基的的不配对几率在此区间。一个比较简单直接的方法是增加[0.1, 0.9]区间对损失的贡献，比如从损失中减去(几率-0.5)的平方, 可是这方面的尝试并没有减小RMSE。我们采用的交叉熵能增加在0/1附近的梯度，也没有能够减小RMSE。另外一个明显的问题是RMSE过大，常见的解决方法是增加模型的深度或宽度，我们发现更大的模型能够更好地拟合训练数据（比如模块维度为128层数为3时可达到RMSE < 0.1），可验证数据的RMSE停留在0.2左右。我们认为这是一个典型的过度拟合现象，所以没有采用更大的模型。

<img src="https://ai-studio-static-online.cdn.bcebos.com/15636ffe2dc347328d46774059f356233c842e16bb5447bc86939158b46882ed" width=600/>

图表 9: 模型预测碱基不配对几率分布

在最后递交的预测结果里, 鉴于我们的模型较小, 我们采用了模型平均。在每次训练中我们从训练集中随机选取10%的序列做为验证集并存取验证集最小RMSE的模型参数。多次的训练就得到不同的参数集的同一模型，最后的结果我们平均了三个最好的参数集的预测。可能因为是出于同一模型，模型平均也仅降低RMSE不到0.5%。

## 总结与讨论 ##
本文主要叙述了作者参加螺旋桨RNA结构预测竞赛的过程, 尤其关于探索的很多走的通和走不通的方向, 希望对感兴趣的读者有些借鉴. 基因疗法团队是深度学习的初学者, 一个很大的感触是飞浆深度学习平台的强大功能和极低的学习门槛。我们深度学习框架的搭建绝大部分用的是飞浆平台提供的成熟的库函数，包括用到的TransformerEncoder /LSTM等等只需直接调用。飞浆是我们学习的第一也是唯一的深度学习平台, 这些佐证了飞浆的成熟和易用性。 

如前文所述, RNA碱基不配对几率对科学和医学意义重大, 尤其我们可以说RNA医学已经到来而且必将蓬勃发展. 这次竞赛也正是为了提高对碱基不配对几率预测的精度, 我们在训练中能达到和验证集的RMSE~0.20, 比赛测试集得到的RMSE差一些, ~0.24. 我们猜想一个原因是测试集有31%的序列长度超过训练和验证集中的最长序列长度, 增加了预测难度. 无论是0.20还是0.24, 和实验结果都相距甚远, 还远远达不到实用或和实验媲美的精度, 可提高的空间还很多. 在现有的基础上我们还可以尝试更多的方法来提高预测精度, 下面讨论一些可能的方向.

首先, 我们可以增强输入信息. 比如在LinearFold和LinearPartition的基础上我们可以加入更多计算方法的预测数据. 在RNA结构预测领域有很多可用的软件, 比如基于物理的MFOLD[4], VIENNA RNAFOLD[5], RNASTRUCTURE[6], 基于机器学习的CONTRAFOLD[7], CDPFOLD[8], 和近年来快速发展的深度学习方法SPOT-RNA[9], E2EFOLD[10], MXFOLD2[11], DIRECT[12], RNACONCAT[13], DMFOLD[14]. 

其次, 我们的模型相对较小, 难于捕捉RNA碱基配对所受支配的碱基间的多体相互吸引与竞争, 所以一个提高预测精度的方向是训练更大的模型. 尤其是当RNA序列长度更长, 比如新冠病毒的基因有~30,000个碱基, mRNA疫苗也有几千个碱基, 并且其碱基经过大量的化学修饰, 预测其结构更是难上加难. 要能够预测更长更复杂的RNA, 更大的模型几乎是必须的. 比如图10展示了模型和标注之间的对比, 上下两组的长度分别为110和480, 我们可以看到对较短的序列预测准确度很高, 可是对长序列的预测差距很大. 更大的模型就需要更多的更全面的训练数据, 尤其对于有化学修饰的碱基, 实验测量很有可能是一个瓶颈. 需要我们另辟蹊径, 比如用模拟的方法得到更多的数据. 

![](https://ai-studio-static-online.cdn.bcebos.com/a35b6dd3835f4c8fa35a3f277a6a9b279f721edf220a4fb694723144f3d2d8e3)
![](https://ai-studio-static-online.cdn.bcebos.com/cdc9f78781f64feb8e838fe9a2b2f1ebbf61c233d67d46c58f7e8524459c5d56)

图表 10: 模型预测碱基不配对几率和标注的对比, 上排两图中的RNA范例长度为110, 下排长度为480. 左列两图展示了标注(蓝),预测(红)和它们的差(绿)对序列位置(X), 右列两图展示了预测(Y)对标注(X).

最后, 从纯科学角度上, 一个理想的模型只需要RNA序列信息的就能准确预测碱基不配对几率和其它的结构信息. 这并不是可望而不可及的, 在现有的方法里, 比如百度螺旋桨的LinearFold/LinearPartition和前面提到的第一个端到端的深度学习方法SPOT-RNA都只需要序列就可以非常好的预测RNA的二级结构. 当然从实用角度上看, 即使一个方法需要综合很多方面的信息(比如需要聚合很多现有软件的预测结果), 如果能较快的准确预测RNA结构, 那也已足够. 条条大路通罗马, 很多道路需要摸索, 每条路都需要很多人的不懈努力.

### 引用文献 ###
1.	Huang L, Zhang H, Deng D, Zhao K, Liu K, Hendrix DA, Mathews DH. LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search. Bioinformatics. 2019;35(14):i295-i304. 
2.	Zhang H, Zhang L, Mathews DH, Huang L. LinearPartition: linear-time approximation of RNA folding partition function and base-pairing probabilities. Bioinformatics. 2020;36(Supplement_1):i258-i67. 
3.	Zhang H, Zhang L, Li Z, Liu K, Liu B, Mathews DH, Huang L. LinearDesign: Efficient Algorithms for Optimized mRNA Sequence Design2020 April 01, 2020:[arXiv:2004.10177 p.]. Available from: https://ui.adsabs.harvard.edu/abs/2020arXiv200410177Z.
4.	Zuker M. Mfold web server for nucleic acid folding and hybridization prediction. Nucleic Acids Research. 2003;31(13):3406-15. 
5.	Lorenz R, Bernhart SH, Höner zu Siederdissen C, Tafer H, Flamm C, Stadler PF, Hofacker IL. ViennaRNA Package 2.0. Algorithms for Molecular Biology. 2011;6(1):26. 
6.	Reuter JS, Mathews DH. RNAstructure: software for RNA secondary structure prediction and analysis. BMC Bioinformatics. 2010;11(1):129. 
7.	Do CB, Woods DA, Batzoglou S. CONTRAfold: RNA secondary structure prediction without physics-based models. Bioinformatics. 2006;22(14):e90-e8. 
8.	Zhang H, Zhang C, Li Z, Li C, Wei X, Zhang B, Liu Y. A New Method of RNA Secondary Structure Prediction Based on Convolutional Neural Network and Dynamic Programming. Frontiers in Genetics. 2019;10(467). 
9.	Singh J, Hanson J, Paliwal K, Zhou Y. RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning. Nature Communications. 2019;10(1):5407. 
10.	Chen X, Li Y, Umarov R, Gao X, Song L. RNA Secondary Structure Prediction By Learning Unrolled Algorithms2020 February 01, 2020:[arXiv:2002.05810 p.]. Available from: https://ui.adsabs.harvard.edu/abs/2020arXiv200205810C.
11.	Sato K, Akiyama M, Sakakibara Y. RNA secondary structure prediction using deep learning with thermodynamic integration. Nature Communications. 2021;12(1):941. 
12.	Jian Y, Wang X, Qiu J, Wang H, Liu Z, Zhao Y, Zeng C. DIRECT: RNA contact predictions by integrating structural patterns. BMC Bioinformatics. 2019;20(1):497. 
13.	Sun S, Wang W, Peng Z, Yang J. RNA inter-nucleotide 3D closeness prediction by deep residual neural networks. Bioinformatics. 2020;37(8):1093-8. 
14.	Wang L, Liu Y, Zhong X, Liu H, Lu C, Li C, Zhang H. DMfold: A Novel Method to Predict RNA Secondary Structure With Pseudoknots Based on Deep Learning and Improved Base Pair Maximization Principle. Frontiers in Genetics. 2019;10(143). 


## 附
### 使用方式

详细使用方法请参照fly.ipynb中的文档内容.

A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/1479469)

     需要拷贝work目录和fly.ipynb 保持原有目录结构，即可运行fly.ipynb

B：在本机运行方法如下：

    1) 克隆 github repo 到本地目录

    2) 安装所需函数库 (参照requirements.txt)
    
    3) 运行 fly.ipynb

### 运行示范

```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
# !mkdir /home/aistudio/external-libraries
# !pip install colorlog -t /home/aistudio/external-libraries
```


```python
# 程序在/work/code目录下， 需先加入路径
import sys 
sys.path.append('/home/aistudio/work/code')
sys.path.append('/home/aistudio/external-libraries')
# fly_paddle是唯一需要直接调用的模块
# fly_paddle is the only module required for interactive sessions
import fly_paddle as fp

# args包括几乎所有需要的参数， 贯穿于几乎所有的程序调用中
# args由fp.parse_args2()根据任务初始化, 要用到的任务包括： ‘train', 'validate', 'predict'
# args is a structure storing most (if not all) parameters and used for most function calls.
# args is initialized by fp.parse_args2(), depending on the specific task, such as "train", "validate" "predict"
args, _ = fp.parse_args2('train')
print(fp.gwio.json_str(vars(args)))
# 注： 根据不同的网络等等需要， args可能包含一些用不到的参数
# Attention: some parameters in args may not be used depending on the network etc.
```

    {
       "action": "train",
       "argv": "-h",
       "verbose": 1,
       "resume": false,
       "load_dir": null,
       "save_dir": null,
       "save_level": 2,
       "save_grpby": ["epoch", "batch"],
       "log": "fly_paddle-Jun19.log",
       "data_args": "======= data args =======",
       "data_dir": "data",
       "data_name": "predict",
       "data_suffix": ".pkl",
       "data_size": 0,
       "test_size": 0.1,
       "split_seed": null,
       "input_genre": "Seq",
       "input_fmt": "NLC",
       "seq_length": [0, 512, -1],
       "residue_fmt": "vector",
       "residue_nn": 0,
       "residue_dbn": false,
       "residue_attr": false,
       "residue_extra": false,
       "label_genre": "upp",
       "label_fmt": "NL",
       "label_tone": "none",
       "label_ntype": 2,
       "label_smooth": false,
       "net_args": "======= net args =======",
       "net_src_file": "/home/aistudio/work/code/paddle_nets.py",
       "net": "lazylinear",
       "resnet": false,
       "act_fn": "relu",
       "norm_fn": "none",
       "norm_axis": -1,
       "dropout": 0.2,
       "feature_dim": 1,
       "embed_dim": 32,
       "embed_num": 1,
       "linear_num": 2,
       "linear_dim": [32],
       "linear_resnet": false,
       "conv1d_num": 1,
       "conv1d_dim": [32],
       "conv1d_resnet": false,
       "conv1d_stride": 1,
       "conv2d_num": 1,
       "conv2d_dim": [32],
       "conv2d_resnet": false,
       "attn_num": 2,
       "attn_nhead": 2,
       "attn_act": "relu",
       "attn_dropout": null,
       "attn_ffdim": 32,
       "attn_ffdropout": null,
       "lstm_num": 2,
       "lstm_dim": [32],
       "lstm_direct": 2,
       "lstm_resnet": false,
       "output_num": 1,
       "output_dim": [32, 32, 2],
       "output_resnet": false,
       "optim_args": "======= optim args =======",
       "optim": "adam",
       "learning_rate": 0.003,
       "beta1": 0.9,
       "beta2": 0.999,
       "epsilon": 1e-08,
       "lr_scheduler": "reduced",
       "lr_factor": 0.9,
       "lr_patience": 10,
       "weight_decay": "none",
       "l1decay": 0.0001,
       "l2decay": 0.0001,
       "train_args": "======= train/loss args =======",
       "batch_size": 4,
       "num_epochs": 777,
       "num_recaps_per_epoch": 30,
       "num_callbacks_per_epoch": 10,
       "loss_fn": ["mse"],
       "loss_fn_scale": [1],
       "loss_sqrt": false,
       "loss_padding": false,
       "validate_callback": null,
       "trainloss_rdiff": 0.001,
       "validloss_rdiff": 0.001,
       "trainloss_patience": 11,
       "validloss_patience": 11,
       "mood_args": "======= mood args =======",
       "debug": false,
       "lucky": false,
       "lazy": false,
       "sharp": false,
       "comfort": false,
       "explore": false,
       "exploit": false,
       "diehard": false,
       "tune": false,
       "action_args": "======= action args ======="
    }


    /home/aistudio/work/code/misc.py:52: DeprecationWarning: invalid escape sequence \e
      """ escape char: \033 \e \x1B  """



```python
# 两种更新args的方法： 1） args.update(**dict), 2) args.[key] = value
# Two main ways to update values in args: 1) args.update(**dict), 2) args.[key] = value
args.update(data_dir='work/data', data_name='train', residue_dbn=True, residue_extra=True)

# 网络参数 （net parameters): 
# 网络的设计主要考虑了三个支配RNA碱基配对的因素： 
#    1) 来自于全部序列的排列组合（配分）竞争，用Attention机制来模拟
#    2）来自于线性大分子的一维序列限制， 用LSTM结构来模拟
#    3）来自于局部紧邻碱基的合作（比如，一个孤立的碱基对极不稳定）， 用1D Convolution来模拟
# 所以框架由以上三个模块组成， 并在输入和输出层加了1-3个线性层。除非特意说明， 所有的隐藏层的维度为32.
# 训练中发现高维度和深度的网络并不能给出更好的结果！
# Three main mechanisms directing RNA base pairing are taken into consideration for the 
# design of the network architecture. 
#   1) The combinatorial configurational space of attainable RNA base pairs, approximated by Attention Mechanism
#   2) The quasi-1D nature of unbranched, continuous RNA polymers, approximated by LSTM
#   3) The cooperativity of neighboring bases for stable base pairing, approximated by 1D Convolution
# Hence the neural net comprises of three main building blocks, in addition to linear layers for the input and output. 
# The dimensions of all hidden layers are 32 unless noted otherwise.
# Wider and/or deeper nets gave similar, but no better, performances!
args.net='seq2seq_attnlstmconv1d'  # the net name defined in paddle_nets.py
# 输入模块由一个线性层组成
# The input block is a single linear feedforward layer
args.linear_num = 1 # the number of linear feedforward layers
# 三大处理模块 (the three main data-crunching blocks)
args.attn_num = 1 # the number of transformer encoder layers
args.attn_nhead = 2 # the number of heads (2 chosen to capture paired/unpaired states, naively)
args.lstm_num = 1 # the number of bidirectional lstm layers
args.conv1d_num = 1 # the number of 1D convolution layers
# 输出模块由三个线性层组成， 维度分别为32, 32, 2
# Three linear layers for the final output, with dimensions of 32, 32, and 2, respectively
args.output_dim = [32, 32, 2]
# 如果序列被补长到同一长度， 对归一化的影响不清楚， 所以用batch_size=1
# If sequences are padded to the same length, such padding may interfere with normalization, hence batch_size=1 
args.norm_fn = 'layer' # layer normalization
args.batch_size = 1 # 1 is used in consideration of the layer norm above
# 最后递交用的损失函数选为softmax+bce, 也可以用 softmax+mse, 结果几乎一样
# The submitted results were trained with softmax+bce as the loss function. 
# Essentially the same results were obtained with softmax+mse
args.loss_fn = ['softmax+bce'] # softmax is needed here as the final output has a dimension of 2
args.label_tone = 'soft' # soft label for upp
args.loss_sqrt = True # sqrt(loss) is only necessary for softmax+mse
args.loss_padding = False # exclude padded residues from loss calculation
# 需要运行fp.autoconfig_args()来消除参数的不一致性
# fp.autoconfig_args() needs to be run to resolve inconsistencies between parameters
args = fp.autoconfig_args(args)
```


```python
# 建立和检测模型 （Get and inspect the model）
model = fp.get_model(args)
# 注： 最后的输出矩阵的维度为[N, L, 2]
# Note: the shape of the output is [N, L, 2]
```

    2021-06-19 18:48:33,958 - INFO - Used net definition: [0;39;46m/home/aistudio/work/code/paddle_nets.py[0m
    2021-06-19 18:48:34,041 - INFO - {'total_params': 36418, 'trainable_params': 36418}
    2021-06-19 18:48:34,042 - INFO - Optimizer method: adam
    2021-06-19 18:48:34,042 - INFO -    learning rate: 0.003
    2021-06-19 18:48:34,043 - INFO -     lr_scheduler: reduced
    2021-06-19 18:48:34,043 - INFO -     weight decay: none
    2021-06-19 18:48:34,044 - INFO -          l1decay: 0.0001
    2021-06-19 18:48:34,044 - INFO -          l2decay: 0.0001
    2021-06-19 18:48:34,044 - INFO - Getting loss function: ['softmax+bce']


    -------------------------------------------------------------------------------------------------------------------------------------
          Layer (type)                          Input Shape                                  Output Shape                   Param #    
    =====================================================================================================================================
       MyEmbeddingLayer-1                      [[2, 512, 10]]                                [2, 512, 10]                      0       
            Linear-1                           [[2, 512, 10]]                                [2, 512, 32]                     352      
             ReLU-1                            [[2, 512, 32]]                                [2, 512, 32]                      0       
           LayerNorm-1                         [[2, 512, 32]]                                [2, 512, 32]                     64       
            Dropout-1                          [[2, 512, 32]]                                [2, 512, 32]                      0       
         MyLinearTower-1                       [[2, 512, 10]]                                [2, 512, 32]                      0       
        PositionEncoder-1                      [[2, 512, 32]]                                [2, 512, 32]                      0       
           LayerNorm-2                         [[2, 512, 32]]                                [2, 512, 32]                     64       
            Linear-2                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
            Linear-3                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
            Linear-4                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
            Linear-5                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
      MultiHeadAttention-1    [[2, 512, 32], [2, 512, 32], [2, 512, 32], None]               [2, 512, 32]                      0       
            Dropout-3                          [[2, 512, 32]]                                [2, 512, 32]                      0       
           LayerNorm-3                         [[2, 512, 32]]                                [2, 512, 32]                     64       
            Linear-6                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
            Dropout-2                          [[2, 512, 32]]                                [2, 512, 32]                      0       
            Linear-7                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
            Dropout-4                          [[2, 512, 32]]                                [2, 512, 32]                      0       
    TransformerEncoderLayer-1                  [[2, 512, 32]]                                [2, 512, 32]                      0       
      TransformerEncoder-1                  [[2, 512, 32], None]                             [2, 512, 32]                      0       
          MyAttnTower-1                        [[2, 512, 32]]                                [2, 512, 32]                      0       
             LSTM-1                            [[2, 512, 32]]                  [[2, 512, 64], [[2, 2, 32], [2, 2, 32]]]     16,896     
          MyLSTMTower-1                        [[2, 512, 32]]                                [2, 512, 64]                      0       
            Conv1D-1                           [[2, 512, 64]]                                [2, 512, 32]                   10,272     
             ReLU-2                            [[2, 512, 32]]                                [2, 512, 32]                      0       
           LayerNorm-4                         [[2, 512, 32]]                                [2, 512, 32]                     64       
            Dropout-5                          [[2, 512, 32]]                                [2, 512, 32]                      0       
         MyConv1DTower-1                       [[2, 512, 64]]                                [2, 512, 32]                      0       
            Linear-8                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
             ReLU-3                            [[2, 512, 32]]                                [2, 512, 32]                      0       
           LayerNorm-5                         [[2, 512, 32]]                                [2, 512, 32]                     64       
            Dropout-6                          [[2, 512, 32]]                                [2, 512, 32]                      0       
            Linear-9                           [[2, 512, 32]]                                [2, 512, 32]                    1,056     
             ReLU-4                            [[2, 512, 32]]                                [2, 512, 32]                      0       
           LayerNorm-6                         [[2, 512, 32]]                                [2, 512, 32]                     64       
            Dropout-7                          [[2, 512, 32]]                                [2, 512, 32]                      0       
            Linear-10                          [[2, 512, 32]]                                [2, 512, 2]                      66       
         MyLinearTower-2                       [[2, 512, 32]]                                [2, 512, 2]                       0       
    =====================================================================================================================================
    Total params: 36,418
    Trainable params: 36,418
    Non-trainable params: 0
    -------------------------------------------------------------------------------------------------------------------------------------
    Input size (MB): 0.04
    Forward/backward pass size (MB): 9.73
    Params size (MB): 0.14
    Estimated Total Size (MB): 9.91
    -------------------------------------------------------------------------------------------------------------------------------------
    



```python
# 读取数据. 提供的数据被转换成了一个dict, 存储为pickle文件. 
# 输入矩阵中最后两列的数据为linear_partition_c和linear_partition_v的预测结果
# Load data. The provided data are transfomed into a dict saved into a pickle file
# the last two columns in the input matrix are the predictions of linear_partition_c and linear_partition_v
midata = fp.get_midata(args)
train_data, valid_data = fp.train_test_split(midata, test_size=0.1)
```

    2021-06-19 18:48:34,050 - INFO - Loading data: work/data/train.pkl
    2021-06-19 18:48:34,101 - INFO -    # of data: 5000,  max seqlen: 500, user seq_length: [0, 512, -1]
    2021-06-19 18:48:34,102 - INFO -  residue fmt: vector, nn: 0, dbn: True, attr: False, genre: upp
    2021-06-19 18:48:34,121 - INFO - Selected 5000 data sets with length range: [0, 512, -1]



```python
# 训练模型，最后的loss应该在[0.52, 0.53]区间内. 每epoch里运行10次validation check, 存储最优的.
# 每epoch需要五分钟左右(在CPU上)， 自然结束需要～20个epoch
# Train the model - the final loss should be within 0.52 and 0.53.
# In each epoch, validation check is run 10 times, and the best model is saved.
# It takes about 5 minutes to complete one epoch. 
# A natural stop will go between 20 and 30 epochs
train_loss, valid_loss = fp.train(model, train_data, num_epochs=21, validate_callback = fp.func_partial(fp.validate_in_train, midata=valid_data, save_dir='./'))
# 注1：软标签的情况下不能得到0的交叉熵
# Note1: zero cross-entropy is not possible with soft labels
# 注2: 没有做特别的超参数优化, 基本上是缺省设置. 如前面提到, 更宽和更深的网络没有得到更好的结果.
# 因为train_data and valid_data得到相近的结果, 没有用L1/L2 regularization
# 一个显著的问题是整体收敛过快, 可能是因为网络较小. 后期工作可以调整learning_rate, dropout等等
# Note2: No particular efforts were made towards optimizer tweaking. Default values appeard to work fine.
# As mentioned earlier, wider and deeper nets didn't fare better, so were discarded in later trainings.
# No significant variations were observed between train_data and valid_data, so didn't use L1/L2 regularization.
# One conspicuous issue is that the model converges too quickly, future work may attempt to tune 
# hyperparameters such as learning rate, dropout, etc.
```


```python
ax = train_loss.plot.scatter('ibatch', 'loss')
ax = train_loss.groupby('epoch').mean().plot.scatter('ibatch', 'loss')
```


```python
# 读取最后一个checkpoint目录 (忽略优化器state_dict读取错误)
# Load the last saved earlystop directory （ignore the error in optimizer state_dict loading)
fp.state_dict_load(model, model.validate_hist.saved_dirs[-1])
# 可以改动损失函数，检测mse损失（单个模型最好的结果：～0.20）
# The loss_fn can be changed to softmax+mse to check the mse loss.
# (the best/lowest loss obtained from a single model was ~0.20)
args.loss_fn = ['softmax+mse']
model.loss_fn = fp.get_loss_fn(args)
valid_loss = fp.validate(model, valid_data, verbose=1, batch_size=64) # try a larger batch_size, should make no difference though
```


```python
# 读取预测数据， 存储预测结果
# 提交的结果是平均了三次运行最好checkpoint模型， 它们分别得到了0.24, 0.24, 0.242的sqrt(mse)损失， 平均后得到了0.238
# 可惜前两次的checkpoint没有被保存， 只保存了预测的结果.
# 虽然是同一个网络架构， 因为每次训练的train_test_split是随机的， 模型平均的效果和cross_validate相近
# Load the prediction data, and save the predicted results
# The submitted results are the average of the best checkpoints/earlystops from three independent trainings.
# Unfortunately the checkpoints for the first two were not kept, except for the predicted results。
# As each training randomly splits the train and validation data, this model averaging approximates
# the effect/benefit of cross-validation.
predict_data = fp.get_midata(args, data_name='predict', seq_length=-1)
y_model, std_model = fp.predict(model, predict_data, save_dir='predict.files', batch_size=1)
```


```python
!zip -r9oT predict.files.zip predict.files/
```
