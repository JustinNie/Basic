# 1. 定义

全连接层是把前一层的所有节点和下一层的所有节点相连。而卷积层简化了这一操作，对于后一层的节点来说（卷积操作输出），只有少量的局部前一层的节点与其相连。并且参数对于所有节点都是共享的，也就是说，对后一层的节点而言，他们都是用相同的参数来从前一层提取特征。



# 2. FP

首先定义输入$\boldsymbol X: [N, C, H, W]$，其中$N$是Batch的大小，C是输入的channel数量，H、W分别是输入image的高和宽。卷积核$\boldsymbol W: [OC, C, KH, KW]$，其中OC是输出channel的大小，KH、KW分别是卷积核的高和宽。输出$\boldsymbol Y: [N, OC, OH, OW]$，其中OH、OW分别是输出image的高和宽。
$$
OH = \lfloor \frac {H + 2 * P - KH}{stride}\rfloor + 1, \qquad
OW = \lfloor \frac {W + 2 * P - KW}{stride}\rfloor + 1
$$
kernel的大小都是奇数，P是padding的长度，具体操作就是输出的节点以自己为中心，用kernel和对应输入image位置上的节点相乘得到。

具体在实现中，一般会把kernel展开成OC列，对应每个输出的channel，每一列有$C*KH*KW$个元素，对应每个channel需要相乘的参数。同样会把输入X展开成$N*OH*OW$行，对应N个Batch输出的每一个元素，每一行当然有$C*KH*KW$个元素，对应生成一个输出的元素。具体如下所示：
$$
X: [N, C, H, W] \qquad -> \qquad [N*OH*OW, C*KH*KW]\\
W: [OC, C, KH, KW] \qquad -> \qquad [C*KH*KW, OC] \\
Y = XW: [N*OH*OW, OC] \qquad -> \qquad [N, OC, OH, OW]
$$
重点是kernel把每一列都作为输出一个元素的参数，输入X展开成每一行对应输出元素的元素块。

关于参数，其实还有一个$B: [OC, 1]$，对每个输出的channel有一个偏移量。所以对整个卷积层来说，总参数个数为：
$$
OC * (C * KH * KW + 1)
$$
如果是采用全连接层连接，那么参数个数就会飙升到（假设输出大小不变）：
$$
C*H*W + 1
$$


# 3. BP

从矩阵相乘角度来看：$Y=XW$，那么，自然（有一点问题，需要根据损失函数的求导法则进行转置）：

若已知损失函数对输出的导数为：$\frac {\partial l}{Y}_{(N*OH*OW, OC)}$，则：
$$
\frac {\partial l}{\partial W}_{(C*KH*KW, OC)}=X^T_{(C*KH*KW, N*OH*OW)}\frac {\partial l}{\partial Y} _{(N*OH*OW, OC)}\\
\frac {\partial l}{\partial X}_{(N*OH*OW, C*KH*KW)}= \frac {\partial l}{\partial Y} _{(N*OH*OW, OC)} W^T_{(OC, C*KH*KW)}
$$
然后再根据原有的格式重新组织即可（即转换维度，该相加的项相加）。

关于一般的标量对矩阵求导的传播在MLP里有一定的规律总结，适合于标量对矩阵或者向量求导的传递。





# 4. pooling层

## 4.1 定义

pooling是仿照人的视觉系统对卷积后的特征进行降维处理，用更高层次的特征抽象。常见的pooling方式有Average Pooling和Max Pooling。

Max Pooling通常可以保留边缘纹理信息，可以减小估计值方差，保留较强的激活值，但容易过拟合；而Average则会见小偏差，弱化激活值。

池化操作可以重叠也可以不重叠。

## 4.2 FP & BP

前向传播的时候，本区域内的值去均值或者取最大值，合并为一个值。反向传播的时候，对于Average Pooling，把输出的特征图展开成原来大小的特征图，对每个位置的梯度乘以Pooling区域的大小；对于Max Pooling而言，对应Max位置上梯度直接传播，其他位置梯度为0。



#5. 相关问题

* 卷积操作为什么要参数共享？
  * 减少参数的数量。
  * 不同区域的特征分布是相似的，所以可以用相同参数的卷积核来提取。
* 为什么要pooling？
  * 减小空间大小，一定程度上完成数据降维，提取特征。
  * 减少网络参数，防止过拟合。
  * 因为滑动的时候具有大量重叠区域，pooling可以消除这些冗余，并且忽略局部微小形变。
  * Pooling可以保持特征的位置和旋转不变性，如果位置信息非常重要，则不建议用。
  * 有一个缺点是忽略了强特征的多次出现。
  * 如果位置信息重要，可以考虑引入位置相关的Top-K值。

























































































