# 2.24

## 深度学习介绍



![image-20250224132410620](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250224132410620.png)

自然语言处理——机器翻译（主要是感知）

计算机视觉——推理（像素，符号学x） 

有什么用——ex.图片分类，物体检测（是个什么东西）分割（每个像素属于哪里），样式迁移（滤镜），人脸合成，文字生成文字/图片，无人驾驶

案例：广告点击

触发、点击率预估、排序（点击率×价格）

预测与训练 特征提取（广告主，描述，图片），通过模型预测

把广告和用户点击存下来作为数据来训练模型



深度学习的应用会有三类人：

![image-20250224133216815](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250224133216815.png)

（生信专业的人干的就是“领域专家”的工作——了解生物知识和基本的深度学习原理）

### Q&A

可解释性？黑盒（深度学习这一块做的不好），有效性和可解释性是不一样的

（弹幕：就跟中医一样？不好解释原理但是有用）

领域专家：“甲方”，知道特征提需求的人

数据科学家：将实际问题变为模型 AI专家：把模型做好（二者有重合）

如何找paper？（之后会讲？）

## 安装

`pip install xxx` ：一个个装，可以用镜像加速 `-i`+链接

`jupyter notebook `：启动（点浏览器链接）

对于zip压缩文件：

`sudo apt install zip` 安装解压功能

`unzip xxx.zip` 解压

 ## 数据操作+数据预处理

### 数据操作原理

N维数组是机器学习和神经网络的主要数据结构

ex. 0维：标量   1维：向量     2维：矩阵

（3d：一张RGB图片（宽x高x3（R/G/B，通道）），4d：一批RGB图片（批量大小x宽x高x通道），5d：一批视频（批量大小x时长x....）

创建数组：形状（如3x4），数据类型（整数、浮点数），数据值（全0 or随机数（给定分布））

访问元素：（numpy数组）

![image-20250224142043968](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250224142043968.png)

（“一列”写反了）

`a:b`：[a,b) 左闭右开，不写就是到头

`::x`：跳着选，每x个选一个

### 数据操作代码实现

`import torch` （虽然叫pytorch但是导入的是torch，包括前面也是	`pip install torch`）



张量（tensor）表示一个数值组成的数组，可能多个维度

如:`x=torch.arange(12)`（0到11）

`x.shape`访问形状（1x12张量），`x.numel`（numbers of elements）元素总数（此处输出12）

改变形状而不改变数量和元素值，用reshape函数：`X=x.reshape(3,4)`（reshape(2,5)这种会报错）

zeros，ones

可以用列表/嵌套列表来赋予确定值 `x=torch.tensor([[1,1,4],[5,1,4]])` 

对于张量，标准的算术运算符（+ - * / **）都是按元素运算

甚至可以指数运算e的x次方 `torch.exp(x)`



连接：`torch.cat(X,Y),dim=0`，在第0维连接（6行4列），dim=1则为3行8列（2维时，-2和0效果相同，-1和1效果相同）

通过逻辑运算符构建二元张量，如`X==Y`，值全为true/false，判断对应元素相不相等

对所有元素求和：`X.sum()`，生成只有一个元素的张量

**形状不同：广播机制**

ex.`a=([0],[1],[2])` （3行1列）`b=([0,1]`) （1行2列）→复制成3行2列

`a+b=([0,0],[1,1],[2,2])+([0,1],[0,1],[0,1])=([0,1],[1,2],[2,3])`

（必须保证维度相同+行/列数整数倍）

（防止出错）



访问：ex.[-1]最后一个 ，[1:3]，第1,2个（左闭右开）

指定索引赋值：ex.`X[1,2]=9`

多个元素（区域）赋值：ex.`X[0:2,:]=12`（第0,1行的所有元素都变成12）

**内存问题**

`id(x)`：python里面每个object都有个唯一的标识号

`Y=Y+X`前后，Y的id不再相同

原位操作：`Y[:]=Y+X`（or `Y+=X`），id不改变

(`Z=torch.zeros_like(Y)`，生成和Y形状相同的全0张量)



最常见的为numpy——torch张量可以由numpy转化

`A=X.numpy()` 	`B=torch.tensor(A)`

大小为1的张量可以转为python标量——

`a ` `a.item` `float(a)` `int(a)`

### 数据预处理

简单的预处理实例：房子的房间数/铺路情况/价格

原始数据集写入csv文件

用pandas加载原始数据集`data=pd.read_csv(file)`

 对于缺失的数据：**插值** or **删除**

输入特征/输出  iloc=index location，取子矩阵（？）

`imputs=imputs.fillna(inputs.mean())` 

对于数值变量，缺失值用平均值代替

`imputs=pd.get_dmmies(inputs,dummy_na=true)`

（变成了一堆01列，表示具备or不具备这个特征）

非数值变量：将“缺失”视为一个类别，字符串转化为类型变量

将csv文件转化为张量格式（`torch.tensor(inputs.values)`（output同理））

数据预处理完成

### Q&A

reshape和view的区别→`b=a.reshape(x,x)`之后创造了a的view，b改了之后a也改了（很奇怪啊.jpg）

数组计算跟不上了怎么办？→学一下numpy

怎样快速区分维度？→`a.shape()` 一下就看出来多长了

pytorch的tensor和numpy的array有区别吗？还是有的（）（底层架构啊什么的 ）（？）

概念上的话：tensor是数学概念，array是计算机概念

新分配了Y的内存，原先的会自动释放吗？→会的

# 3.3

## 线性代数

（数学上的意义，背景入门）

标量

- 简单操作，运算
- 长度，|a|，|a+b|<=|a|+|b|，|ab|=|a||b|

向量

- 简单操作，加法，数乘，函数（c=sin a→ci=sin ai）
- 长度：||a||2=（每一维度平方平均值） ||a+b|| ||αb||同标量

“parallel for"

点乘，正交

矩阵加法，数乘，函数

矩阵乘法

矩阵乘向量——”扭曲一个空间“

范数“

c=Ab→||c||<=||A|| ||b||

矩阵范数：最小的满足上面公式的值

Frobenius范数：
$$
||A||_{Frob}=(\sum_{ij}A_{ij}^2)^{\frac{1}{2}}
$$
特殊矩阵：对称。反对称，正定，正交，置换矩阵

特征值 特征向量

（特征向量：不被矩阵的“扭曲空间”改变方向的向量）

### 代码实现

标量：只有一个元素的张量（`torch.tensor`）

可以进行运算

向量：标量值的列表（如`torch.arange`）

张量的索引来访问任一元素（注意0/1开头）

len，shape（一维张量只有一个元素）

生成矩阵 `A=torch.arange(20).reshape(4,5)`

矩阵转置 `A.T`，对称矩阵`A==A.T`

可以构建更多维度的张量

 相同形状张量做二元运算，结果的形状相同

复制：`B=A.clone()`，`B=A`直接把地址复制了

按元素乘法：哈达玛积`A*B`

#### 求和

所有元素的和`A.sum()`，学过了

指定维度求和（`A.sum(axis=xxx)`）（对谁求和哪个维度就没了）

平均值：`A.mean()` or`A.sum()/A.numel()`，平均值也可以按维度

 保持**维度数**不变：`A,sun(axis=1,keepdims=True)`

（比如：不keep——[1,2,3,4,5]（size=[5]) keep——[[1],[2],[3],[4],[5]] (size=[1.5]) ）



好处：可以采用广播机制让每个数除以sum

对某维度累积求和：`A.cumsum(axis=0)`（类似前缀和）

#### 乘法

向量点乘：`torch.dot(x,y)`

矩阵乘向量：`torch.mv(A,x)`（mv：matrix×vector）

矩阵乘矩阵：`torch.mm(A,B)`

范数：`torch.norm(a) `（勾股定理——l2范数，同时也是矩阵的F范数）

l1范数（绝对值求和）：`torch.abs(x).sum()`

### Q&A

负面影响？稀疏？

深度学习为什么用张量？深度学习是机器学习的一块，机器学习是“计算机学家对统计的理解”，统计常用张量

copy和clone的区别？clone一定会复制内存，copy分copy和deepcopy的区别

torch不区分行向量和列向量吗？一维张量一定是行向量，列向量是nx1的矩阵（二维的）

机器学习的向量就是多维数组，跟数学张量概念不同（数学概念暂时不用管它.jpg）

用什么语言？就跟学车一样，车只是工具，说不定过五年就用别的了呢（）

（陈玉泉：学计算机的好处就是无论你多大都可以跟其他人站在同一起跑线上）

## 矩阵计算

求导 

标量：（高中数学——基本初等函数，导数的运算，导数的几何意义）

亚导数：拓展到不可导的函数（如y=|x|在0处，x=0时候在[-1,1]之间去任意的值，符号和偏导一样）

将导数拓展到向量——

（梯度）



![image-20250303151350412](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303151350412.png)

（列向量→行向量）

![image-20250303151746313](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303151746313.png)

（列向量→列向量）

![image-20250303151832294](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303151832294.png)



（Jacobi矩阵）

再拓展——”矩阵对矩阵求导“？

![image-20250303152205565](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303152205565.png)

 （被挡住的是个四维张量）

附：向量/矩阵的基本求导法则示例

<img src="C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303154200645.png" alt="image-20250303154200645" style="zoom:50%;" />

<img src="C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303154200645.png" alt="image-20250303154200645" style="zoom:50%;" />

## 自动求导

链式法则拓展到向量和矩阵，仍然成立

（ex.线性回归）

 ![image-20250303154534044](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303154534044.png)

自动求导：在指定值上的导数（区别于符号求导/数值求导）

计算图：分解成操作子→无环图（复合函数的过程）

显式构造：Tensorflow

隐式构造：Pytorch

计算方法：正向累积/反向累积（反向传递） 

复杂度：时间复杂度O(n)，n为操作子个数（正向反向类似）

空间复杂度：

### 代码实现

存储梯度：`x.required_grad_(True)`

定义函数y（如`y=torch.dot(x,x)`）

调用反向求导 `y.backward()`

结果存在 `x.grad`里面

梯度清零：`x.grad.zero_()`

把算出来的函数当常数：`u=y.detach()`

计算图需要通过python控制（如if,for,while……），也可以实现求导

### Q&A

为啥显式构造和隐式构造差不多？（类似于用py/数学实现一个函数）

需要正向反向都算一遍吗？用神经网络求梯度的时候，需要的 

（正向写公式，反向求导？）

为什么pytorch会默认累计梯度？（算不下可以拆开）

为什么一般对标量求导（如loss）？为了防止维度扩得太大

（个人理解：正向来构造计算图，反向来求梯度）

## 线性回归

”机器学习中最基础的模型“

应用背景：如何在美国买房（竞拍） 

房价预测：价格-时间关系（很重要，这是真钱！）

模型：与卧室个数/卫生间个数/居住面积有关，加权和，即y=w1x1+w2x2+w3x3+b



抽象模型：

![image-20250303170358860](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303170358860.png)

（线性模型可以看成**单层神经网络**）

![image-20250303170501299](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250303170501299.png)

首次提到“神经网络”——起源于神经科学

衡量预估质量：

比较真实值和预估值，例如房屋售价和估价

平方损失：l（y,y尖)=1/2 (y-y尖)^2
$$
l(y,\hat y)=\frac{1}{2}(y-\hat y)^2
$$
训练数据：

收集数据点决定参数值（权重和偏差），例如过去六个月买的房子各个参数和成交价

这些数据被称之为训练数据，越多越好

（如果数据不够怎么办，会有算法）

如果有n个样本，那这样的话输入X是一个n个行向量拼成的矩阵，Y是nx1的列向量



参数学习：

损失函数：
$$
l(\bold X,\bold Y,\bold w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-<\bold x_i,\bold w>-b)^2=\frac{1}{2n}||\bold y-\bold{Xw}-b||^2
$$
最小化损失：
$$
\bold w^*,b^*=\arg\min_{w,b}l(\bold X,\bold Y,\bold w,b)
$$
argmin：函数值取最小时自变量的值



由于线性模型比较简单，因此具有显式解：

(其它用到神经网络的一般都是NP的)

为了方便，先将偏差加入权重（X加一列变成[X,1]，w加一行变成[w,b]T）

损失函数变成：
$$
l(\bold X,\bold Y,\bold w)=\frac{1}{2n}\|\bold y-\bold{Xw}\|^2
$$
求极值（上节课的复合函数求导）
$$
\frac{\part}{\part \bold w}l(\bold X,\bold Y,\bold w)=\frac{1}{n}(\bold y-\bold X\bold w)^T\bold X=0
$$
于是：
$$
\bold y^T\bold X-\bold w^T\bold X^T\bold X=0
\\ \bold w^T=\bold y^T\bold X(\bold X^T\bold X)^{-1}=\bold y^T\bold X\bold X^{-1}(\bold X^{-1})^T
\\ \bold w^*=(\bold X^T\bold X)^{-1}\bold X^T\bold y
\\ \bold w^*=\bold X^{-1}\bold y(?)
$$
总结：

- n维输入的加权+外加偏差

- 使用平方损失衡量差异

- 线性回归有显式解

- 线性回归可以看成单层神经网络

# 3.10

## 基本优化算法——梯度下降

没有显式解怎么办？随便找一个值，一步一步靠过去：

挑选初始值w0，之后迭代：
$$
\mathbf{w}_t=\mathbf{w}_{t-1}-\eta\frac{\part l}{\part\mathbf{w}_{t-1}}
$$
其中η为学习率（“每一步走多远”），l是损失函数，那个偏导数是梯度（“上升最快的方向”），负梯度就是下降最快的方向

每次沿着下山最快的方向走，就能走到最低点（因为我们要的是最优解，也就是损失函数的最小值）

学习率是步长的”超参数“（人为指定）

学习率的选择：不能太小（太慢，计算梯度代价很高）也不能太大（不准确）

![image-20250310135526328](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250310135526328.png)

 

小批量随机梯度下降：

因为整个训练集计算梯度“太贵了”（几分钟甚至几小时）

→随机采样b个样本来近似损失，即：
$$
l=\frac{1}{b}\sum_{i \in I_b} l(\mathbf{x_i},y_i,\mathbf{w})
$$
b为批量大小（batch size），另一个重要的超参数

同样，b的选取不能太小也不能太大

Summary：

- 梯度下降通过不断沿着梯度的反方向更新参数求解
- 小批量随机梯度下降是深度学习的默认求解算法
- 两个重要的超参数是批量大小和学习率

## 线性回归的从零开始实现



（导入包：random，torch）

构造数据集：y=Xw+b+e（噪声）

```python
def synthetic_data(w,b,num_examples): #给定w,b和样本个数，生成一个数据集
    X=torch.normal(0,1,(num_exapmles,len(w))) #均值为0，方差为1的正态分布，共生成n行w列
    y=torch.matmul(X,w)+b #y=Xw+b，也可写成 y=torch.mv(X,w)+b
    y+=torch.normal(0,0.01,y.shape) #加噪声
    return X,y.reshape((-1,1)) #行数-1表示由pytorch自动推断，列数为1，也可写成y.reshape((num_examples, 1))
```

```python
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=sythetic_data(true_w,true_b,1000)
#参数w=[2,-3.4]^T，b=4.2，生成1000个样本
```

数据集画图：

```python
d2l.set_figsize()
#画出feature(X)的第一列和label(Y)的图
#需要detach到numpy里面（先这样理解）
d2l.plt.scatter(features[:,(1)].detach.numpy(),labels.detach.numpy(),1)
#可以看出来两者有明显线性相关关系
```

读取小批量：

```python
def data_iter(batch_size,features,labels):
    num_examples=len(features) #把这个再搞出来（？）
    indices=list(range(num_examples)) #搞一个0-num-1的list，indices是index（下标）的复数
    random.shuffle(indices)#随机打乱
    for i in range(0,num_examples,batch_size): #每隔batch_size跳一个
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])#然后把中间的batch_size个样本拿出来，min是为了防止越界，如果没那么多就拿到底
        yield features[batch_indices],lables[batch_indices] #返回刚才生成的下标所对应的值，yield相当于有了迭代功能的return（返回之后记住在哪返回的下次从这个地方继续）
        
```

```python
batch_size=10
for X,y in data_iter(batch_size,features,labels):
    print.... #相当于取出了10个样本
    break
```

定义初始化模型参数：

```python
#也就是指定梯度下降算法里的w0
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch,zeros(1,requires_grad=True)
```

定义模型和损失函数：

```python
def linreg(X,w,b): #给出X,w,b返回线性回归的预测值
    return torch.matmul(X,w)+b
def squared_loss(y_hat,y): #y_hat是预测值，y是真实值
    return (y_hat-y,reshape(y_hat.shape))**2/2 #根据损失函数定义，reshape是为了把行向量or列向量统一起来，
#注意这里loss还是一个向量（用的是按元素平方，等会在训练的时候加起来）还没有求均值（等会在梯度下降里面求）
```

定义优化算法：

```python
def sgd(params,lr,batch_size):#param是所有参数（w和b），lr是学习率（learning rate）
    with torch.no_grad(): #更新的时候不用计算梯度
        for param in params: #对于每一个参数
            param-=lr*param.grad/batch_size #用那个公式，这里求均值
            param.grad.zero_()#梯度清零，取消累积
```

指定超参数和模型：

```python
lr=0.03 #指定学习率（每一步走多少）
num_epochs=3 #将数据扫几遍（迭代周期）（走几步）
net=linreg #模型指定为线性回归
loss=suqre_loss #损失函数指定为均方损失
```

开始训练！

```python
for epoch in range(num_epochs):#对数据扫一遍
    for X,y in data_iter(batch_size,features,labels):#拿出一个样本
        l=loss(net(X,w,b),y)#预测并计算损失函数
        l.sum().backward()#在这里加起来，并且反向求导算梯度
        sgd([w,b],lr,batch_size)#梯度下降算法（其实不严谨，因为最后一段可能不全）
        #注意这个地方，结合上节课自动求导学的，y（此处为l）的梯度会存在x.grad（此处为[w,b]，也就是函数里的params，所以要这样写）
        with torch.no_grad():
            train_l=loss(net(features,w,b),labels)#评估模型：计算训练后的损失函数，也可以比较w和b的误差      
```

## 线性回归简洁实现

多import一个data

第一步生成feature和label还是完全一样的



调用框架中现有的API读取数据:

（Application Programming Interface，应用程序编程接口，是一组预先定义的函数等）

```python
def load_array(data_arrays,batch_size,is_train=True):
    #TensorDataset:用于将多个tensor合并在一起
    dataset=data.TensorDataset(*data_arrays)# *表示给元组解包，可以用"feature,label"代替
    #DataLoader:用于批量加载数据，并支持分批和随机打乱等数据处理功能
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

```

```python
batch_size=10
data_iter=load_arraya((features,labels),batch_size)
next(iter(data_iter)) #next得到当前的下一个batch的X和y
```

使用框架预定义好的层

```python
from torch import nn #nn=neural network
net=nn.Sequential(nn.linear(2,1))#线性回归，输入2维，输出1维
#Sequential可以理解为"list of layers"
#由于线性回归只有一层，也可以不用Sequential
```

初始化模型参数：

```python
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0) #bias——偏差，即上文中的b
#net[0]表示net这个Sequential的第一层
#normal和fill都是填充的，和上面代码一样，w0~N(0,0.01),b=0
```

指定损失函数和sgd：

```python
loss=nn.MSELoss() #均方损失
trainer=torch.optim.SGD(net,parameters(),lr=0.03)
```

训练模型（与之前一样）：

```python
num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y) #net把X的模型参数包装进去（？）
        trainer.zero_grad() #优化器梯度清零
        l.backward() #和之前一样，反向求导，这里MSELoss已经帮你求了sum和均值
        trainer.step() #用sgd进行模型更新
    l=loss(net(features),labels)    #评估用的，之后print
```

（要点：和前面完全等价，只不过就是一点一点调用封装包，不要被吓到）

（线性回归虽然简单，但是麻雀虽小也能烧烤，包含了训练一个神经网络的所有基本框架，之后所有的神经网络都会用类似的框架）

### Q&A

为什么用平方而不是绝对值？（其实差别不大，主要是因为绝对值不可导）

损失为什么要求平均？（batch_size一样的话其实不求也没事）

损失函数是否都是mse？基本都是

误差反馈

用n-1替代n？（无偏估计）（也行）

怎么找到合适的学习率？（之后会讲，也可以找一个学习率不依赖的方法）

batch_size太小是否会影响结果？

加噪音防止过拟合？ 

detach()是干啥的？不求梯度，当常数（？）

indices用list行吗？老师说不行但实际可以（）

用生成器相比return什么优势？

如果不是整数倍怎么办？最常见：设置合理 也可以删掉/从下一个epoch补过来

学习率衰减？

收敛如何判断？目测/验证集，epoch数量按直觉选（）

为什么SGD？好多问题没有显式解（NP问题）

w为什么要随机初始化？固定也行

为什么会有nan？不可导

一定要手动设置初始值吗？不一定

## Softmax回归

（机器学习中另一个经典且重要的模型）

（虽然他叫回归，但实际上是个分类问题）

（回归——估计一个连续值；分类——预测一个离散类别）

ex.手写数字识别（10类），自然物体分类

蛋白质分为28类



从回归到分类：

![image-20250310163950213](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250310163950213.png)

 

对类别进行编码：y=[y1,y2,...,yn]T，如果是第i类那i为1其他为0（这样就转化为回归了）

还是采用均方损失训练，最终结果=argmax oi（预测结果为“输出”最大的那一类）

要求对正确类y的置信度（oy）远大于其他类（oi）的置信度（大于某个阈值）

把o转化为输出概率向量：（引入softmax函数，指数函数的目的是为了让每个分量非负）
$$
\hat{\mathbf{y}}=\mathrm{softmax}(\mathbf{o})
\\
\hat{y_i}=\mathrm{softmax}(o_i)=\dfrac{e^{o_i}}{\sum e^{o_k}}
$$
由于两个都是概率，可以用y和y尖的“差别”来衡量损失

概率的区别往往采用交叉熵衡量：
$$
H(\mathbf{p},\mathbf{q})=\sum_i -p_i \log q_i
\\
l(\mathbf{y},\hat{\mathbf{y}})=-\sum_i y_i \log \hat{y_i}
$$
交叉熵的梯度就是真实概率和预测概率的差（对数先假设以e为底，反正就只差个常数）：
$$
l(\mathbf{y},\hat{\mathbf{y}})=-\sum_i y_i \ln\dfrac{e^{o_i}}{\sum e^{o_k}}=-\sum_i y_i(o_i- \ln{\sum_k e^{o_k}})
    \\ \dfrac{\part l}{\part o_i}=-y_i+\sum_i y_i \cdot \dfrac{\part}{\part o_i}(\ln{\sum_k e^{o_k}}) 
  \\ \dfrac{\part}{\part o_i}(\ln{\sum_k e^{o_k}})=\dfrac{1}{\sum e^{o_k}}\cdot e^{o_i}=\mathrm{softmax}(o_i) 
  
  \\又由于\sum_i y_i=1（概率），因此 \dfrac{\part l}{\part o_i}=-y_i+\mathrm{softmax}(o_i)=\hat{y_i}-y_i
$$


Summary：

- softmax回归是一个多类分类模型-

- 使用softmax函数（操作子）可以得到每一个类的预测置信度
- 用交叉熵来衡量预测概率和真实概率的区别

## 损失函数

3个常用的损失函数

1. 用过的，均方损失 L2 Loss
   $$
   l(y,\hat y)=\frac{1}{2} (y-\hat y)^2
   $$

2. L1 Loss，改成绝对值

   (特征：梯度稳定，缺点：零点处不可导)
   $$
   l(y,\hat y)=|y-\hat y|
   $$
   
3. Huber's Robust Loss

   （二者结合）
   $$
   l(y,\hat y)=\begin{cases} |y-\hat y|-\frac{1}{2},|y-\hat y|>1\\\frac{1}{2} (y-\hat y)^2,\text{otherwise}\end{cases}
   $$
   

   

# 3.17

## 图片分类数据集

MNIST数据集（手写数字识别）是图像分类的广泛数据集之一

但是太简单了，我们采用相似但是更复杂的数据集fashion-MNIST数据集(里面是衣服，裤子，鞋子……之类的黑白图片)

```python
import torch
import torchvision #计算机视觉相关
from torchutils import data
from torchvision import transforms 
```

将图片数据下载下来并且转换为32位浮点数格式：

```python
trans=transform.ToTensor() #转化为32位浮点数格式的tensor
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True) #下载训练集
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True) #下载测试集（因此train为False）
```

返回文本标签（数字标签和文本标签的批量转化）：

```python
def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trousers','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]
```

展示图片（数据可视化）（略）

指定数据读取进程数（DataLoader中的num_workers参数）

```python
train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=4)
```

整合成一个函数：

```python
def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transform.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize)) #resize参数决定是否/如何调整大小
        trans=transforrms.Compose(trans) #将多个变换（即上文中的ToTensor和（可能的）Resize整合成一个变换）
        #下载训练集和测试集
        mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
        mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
        #返回两个DataLoader
        
        return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers),data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers),)
```



## Softmax回归的从零开始实现

读取数据：

```python
import torch
from Ipython import display
from d2l import torch as d2l
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
```

由于之前图片像素数为28*28=724，因此将图片展平为长度为784的向量

因为数据集共有10个类别，因此输出一个10维向量

```python
num_imputs=784
num_outputs=10
#指定初值（梯度下降算法中的w0）
W=torch.normal(0,0.01,size=(num_imputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)
```

（矩阵求和复习：按维度求和，keepdim）

实现softmax：

```python
def softmax(X): #对n个行向量组成的矩阵的每一行进行softmax
    X_exp=torch.exp(X) 
    partition=X_exp.sum(1,keepdim=True) #每一个行向量的元素的指数和，keepdim保证他仍为n行1列的矩阵（而非1行n列）
    return X_exp/partition #这里运用了广播机制
def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b) #把X改变形状（10*1*28*28→784*10），方便跟W矩阵乘法
```

 对tensor功能的一点补充：

` y=torch.tensor([1,2,3],[4,5,6]),id=torch.tensor([0,2])`

`y[[0,1],id]`输出`[0,6]`，也就是（0,0）位置和（1,2）位置的数据

实现交叉熵损失函数：

```python
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
#y存的应该是对应类别（?）
#由于其他位置的真实概率值都是0，因此只需要计算对应位置的log(y尖)（×1）
```

将预测类别和真实类别进行比较：

```python
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)#找到输出值取最大值的位置，这就是预测的类型
    cmp=y_hat.type(y.dtype)==y
    #type表示该数据本身的类型（如list等），dtype表示该数据里面的元素的类型（如int等）
    #tensor的type函数：如果没有dtype参数时返回数据类型，如果有dype参数则将该数据转化为dtype表示的类型后返回
    return float(cmp.type(y.dtype).sum())
```

（附：利用nn计算任意模型net的准确率：

```python
def evauate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):#判断net是否为toech.nn的类型
        net.eval() #将模型设置为评估模式
    metric=Accumulator(2) #存储正确预测数和预测总数,Accumulator为python中的累加器
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel()) 
    return metric[0]/metric[1]
    
```

单次模型训练：

```python
#这里把手动定义的模型和torch里面内部自带的模型统一起来，为了实现更多功能
#这里及以下的ch3指的是书上“第三章”用的训练函数
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3) #这次要同时返回损失和精度
    for X,y in data_iter:
        y_hat=net(X) #预测值
        l=loss(y_hat,y) #损失函数
        if isinstance(updater,torch.optim.Optimizer):#使用pytorch自带的
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y),accuracy(y_hat,y),y.size.numel())
        else: #使用自定义的（比如上次的sgd）
            l.sum.backward() #参考上次写的sgd，他是个向量还没求和所以要在这里求一下
            updater(X.shape([0]))
            metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
            return metric[0]/metric[2],metric[1]/metric[2]
```



最终的模型训练函数：

```python
def train_ch3(net,train_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs): #训练这么多次
        train_mertics=train_epoch_ch3(net,train_iter,loss,updater)#训练集
        test_acc=evaluate_accuracy(net,test_iter)#测试集
        train_loss,train_acc=train,metrics #保存参数评估用
```

（在后续的学习中，我们写出的训练函数会越来越复杂，实现的功能也会越来越多）

开始训练！

```python
lr=0.1 #learning
def updater(batch_size): #把之前的小批量随机梯度下降算法准备出来
    return d2l.sgd([W,b],lr,batch_size) 
num_epochs=10 #训练10个周期
train_ch3(net,train_iter,cross_entropy,num_epochs,updater)
```

（这里可以看出，我们写过的所有东西都会放到d2l包里面，之后想用可以直接用）

预测：

```python
def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
    	break #取出一组来
    trues=d2l.get_fashion_mnist_labels(y)
    preds=d2l.get_fashion_mnist_lables(net(X).argmax(axis)=1)
```

## Softmax回归的简洁实现

跟上节课一样，通过深度学习框架的高级API让代码更简洁

导入数据同上

构建神经网络结构，与线性回归不同的是，这里有10个维度的输出

而上次那个“展开成784*10”的操作可以用展平层（flatten）实现

```python
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
```

初始化每一层神经网络的w0：

（当然同样的，单对softmax来说只有一层没必要，但是为了以后的幸福……）

```python
def init_weight(m):
    if type(m)==nn.Linear: #对线性层赋予正态分布的权重
        nn.init.normal_(m.weight,std=0.01)
 net.apply(init_weights)       
```

指定损失函数（交叉熵）和优化算法（sgd）

```python
loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
```

调用之前的训练函数：

```python
num_epochs=10
d2l.train_ch3(net,train_iter,loss,num_epochs,trainer)
```

（由于我们刚刚编写的函数统一了自己写的优化和pytorch自带的优化，这里可以直接用）

（于是，我们实现了最简单最基本的分类模型——Softmax回归，之后我们会介绍更多更复杂的分类模型，并且应用于相同的数据集来对比效果）

### Q&A

softmax回归与logistic回归的区别：logistic可以认为是softmax只有两类的特殊情况

为什么用交叉熵？（简单） ylog(y尖)只关心正确类靠不靠谱？不是不关心，编码就是这么定义的（不是0——softlabel？）

方差0.01有讲究吗？其实是有（毕竟是超参数），之后再说

为什么直接在函数里不除以len？防止最后一个取不全的情况

cmp.type(y.dtype)有必要吗？有（吗 ？）

net.eval()有什么用？其实不开也没事，算是一个好习惯，让他先不用算梯度

测试精度先上升再下降是过拟合了吗？可能

自己实现和api谁快？不好说，但api更稳定

## 感知机

~~赶只鸡~~ 感知机——最早的AI模型之一

给定输入x（向量），权重w（向量）和偏移b

当(<w,x>+b)>0输出1，否则输出-1（或0什么的，反正是一个二分类的问题）

 这里输出的是一个“类别”（二分类变量）而非“实数”（线性）或者“概率”（softmax）

训练过程：

![image-20250317171842861](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317171842861.png)

（注意：y×(<w,x>+b)<0，就意味着预测值和真实值不一样→预测错了）

（损失函数中的max对应着伪代码中的if，更新的值就是损失函数的梯度）

收敛定理：

假设数据在半径为r的区域内，并且所有分类都有正确且有一定余量rou

![image-20250317172218542](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317172218542.png)

感知机的问题：

只能产生线性分割面（因此不能拟合XOR函数（同号为-1，异号为1））

怎样解决？→多层感知机

Summary：

- 感知机是一个二分类模型，是最早的AI模型之一
- 它的求解算法等价于使用批量大小为1的梯度下降
- 它不能拟合XOR函数，导致了第一次“AI寒冬”

## 多层感知机

“如何解决XOR问题”？

→分两步学习

![image-20250317173101878](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317173101878.png)

（先学习“蓝线”，再学习“黄线”，再看两个值是否一样，一样为1，不一样为-1）

（存在中间的隐藏层了！隐藏层的大小是另一个超参数） 

![image-20250317173314367](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317173314367.png)

输入层为n维向量，输出层为m维向量（最终再与w内积得到标量），则隐藏层应为m×n矩阵

![image-20250317173600796](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317173600796.png)

sigma是一个非线性函数

为什么需要非线性的激活函数？→如果是线性的，那么两层运算之后的总输出还是线性，多加的这一层白加了！

最简单的激活函数——sigmoid函数（近似sigma，但中间是连续可导的，没那么生硬）：
$$
\sigma (x)=\begin{cases}1,\ x>0\\0,\ x\le 0\end{cases}
\\
\mathrm{sigmoid}(x)=\dfrac{1}{1+e^{-x}}
$$
类似的，tanh也可以当做激活函数，区间为-1到1（sigmoid为0到1）

ReLU激活函数（rectified linear unit） ReLU(x)=max(x,0)  （优点：不用做指数运算，更快）

多类分类的多层感知机——在softmax中加入隐藏层

![image-20250317174812039](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317174812039.png)

更多的隐藏层：

每一个隐藏层都有一个自己的W和b，也都要有对应的激活函数

超参数又变多了——隐藏层数和每层隐藏层的大小

![image-20250317175408220](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250317175408220.png)

（越大，模型就越复杂，可以选择层数少一些但是每一层大一点or层数多一些但是每一层小一些，一般越深的层次大小越小）（”慢慢压缩“）

 （搞AI的都是“调参工程师”？也没什么特殊经验，全靠手感（？））

Summary：

- 多层感知机使用隐藏层和激活函数得到非线性模型（解决XOR问题）

- 常用的激活函数有Sigmoid，tanh和ReLU（因为ReLU比较简单所以用的比较多）

- 使用softmax来处理多类分类

- 超参数为隐藏层数和各个隐藏层的大小

  

## 多层感知机的代码实现



#### 从零开始实现：

```python
import torch
from torch import nn
from d2l import torch as d2l
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size) #依旧使用上次的fashion mnist作为例子
```

实现一个有单隐藏层的多层感知机：

```python
num_inputs,num_outputs,num_hiddens=784,10,256 #隐藏层设置为256个隐藏单元，输入输出仍然是784和10
W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)) #指定他是“神经网络的参数”（据说不这么做也行？）
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True))
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params=[W1,b1,W2,b2]
```

实现模型：

```python
def relu(X):
    a=torch.zeros_like(X)#生成和X一样维度的0矩阵
    return torch.max(X,a)
def net(X):
    X=X.reshape((-1,num_imputs))#784*1
    H=relu(X@W1+b1) #@表示矩阵乘法
    return (H@W2+b2)
loss=nn.CrossEntropyLoss()#仍采用交叉熵损失
```

 训练模型（与softmax的训练过程完全相同）

```python
num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
```

(损失下降了但是精度没有提高很多？)



#### 简洁实现

```python
import torch
from torch import nn
from d2l import torch as d2l

net=nn.Sequential(nn.Flatten,nn.Linear(784,256),nn.Relu(),nn.Linear(256,10)) #展平+784*256+ReLU+256*10

def init_weight(m): #跟上次那个完全相同
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)

batch_size,lr,num_epoches=256,0.1,10
loss=nn.CrossEntropyLoss()
trainer=torch,optim.SGD(net_parameters(),lr=lr)
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
```

### Q&A

## 模型选择

如何选超参数？

ex.预测谁会还贷款，100个申请人其中5人三年内违约了

并且这五个人在面试的时候都穿了蓝色衬衫，所以……？



误差：训练误差/泛化误差

训练误差：在训练数据上的误差

泛化误差：模型在新数据上的误差（更多关心泛化误差）

ex.通过模拟考试成绩预测高考成绩，背答案or认真做

 

如何计算误差？

验证数据集/测试数据集：

验证数据集——评估模型好坏（比如拿出50%的训练数据，注意不要和训练数据混在一起）

测试数据集——只用一次的数据集，比如高考



K-折交叉验证

在没有足够多数据时使用（这是常态）

算法：

将数据分割为K块，for i=1...K，使用第i块作为验证数据集，其余的作为训练数据集，报告K个验证集误差的平均

常用K=5 or 10



Summary：用训练数据集训练模型参数，用验证数据集选择模型超参数，非大数据集通常使用k折交叉验证



过拟合（overfitting）和欠拟合（underfitting）

![image-20250324150507706](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250324150507706.png)

模型容量：拟合各种函数的能力（模型的“复杂程度”）

低容量模型难以拟合训练数据（比如线性回归肯定没法拟合二次曲线），高容量的模型容易记住所有训练数据（但有可能会把噪声拟合进去造成过拟合）



![image-20250324151222133](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250324151222133.png)

（为了让误差下降需要承受一定程度的过拟合，容量首先得足够大）



估计模型容量：

 难以在不同的种类算法之间比较（例如树模型和神经网络）（？）

给定模型种类，将有两个主要因素：参数的个数/参数值的选择范围

 VC维： （VC dimension，名字来源于两个人）

统计学习理论的核心思想

对于一个分类模型，VC等于一个最大的数据集的大小，不管如何给定标号，都存在一个模型来对它完美分类

 

![image-20250324151900158](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250324151900158.png)

VC维可以提供“为什么一个模型好”的理论依据，可以衡量训练误差和泛化误差之间的间隔

但在深度学习中很少使用（衡量不准确，计算困难）



数据复杂度的因素：样本个数，每个样本的元素个数，时间空间结构，多样性等



### 权重衰退（weight decay）

最常见的处理过拟合的方法



使用均方范数作为限制

通过限制参数值的选择范围控制模型容量

**硬性限制**：在||w||方＜θ的限制下让l（损失函数）最小

（通常不限制b，限不限制都差不多）

（更小的θ意味着更强的正则项，如θ=0时w只能为0向量）

**柔性限制**：让l+λ/2 ||w||方最小

（拉格朗日乘数法证明和硬性限制等效）

（超参数λ控制了正则项的重要程度，相当于是“||w||太大的惩罚”）

 ![image-20250324162505251](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250324162505251.png)

（同样沿用梯度下降算法）

#### 代码实现

“最广泛使用的正则化技术之一”

```python
#加载数据集 
n_train,n_test,num_inmputs,batch_size=20,100,200,5 #(故意把训练集选的少一点让他容易过拟合)
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.05 #y=sum(0.01xi)+0.05+正态偏差
train_data=d2l.sythetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)
train_data=d2l.sythetic_data(true_w,true_b,n_test)
train_iter=d2l.load_array(test_data,batch_size,is_train=False)
```

```python
def init_params():#初始化参数
    w=torch.normal(0,1,size=(sum_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return [w,b]
def l2_penalty(w): #计算||w||方/2，把λ写在外面了
    return torch.sum(w.pow(2))/2
```

```python
def train(lambd): #λ作为超参数，由于有个lambda表达式所以变量名用lambd
    w,v=init_param()
    net,loss=lambda X:d2l.linreg(X,w,b),d2l.squared_loss #lambda表达式作为内联函数
    num_epochs,lr=100,0.003
    for epoch in rang(num_epochs):
        for X,y in train_iter:
            with torch.enable_grad():
                l=loss(net(X),y)+lambd*l2_penalty(W) #按照之前讲的写出加了限制的损失函数
            l.sum.backward()
            d2l.sgd([w,b],lr,batch_size)
    print('W的L2范数是：',torch,norm(W).item())
```

简洁实现：torch.optim.SGD里面有“weight_decay”选项

### 丢弃法（Dropout）



动机：一个好的模型需要对输入数据的扰动有很好的抗干扰性（鲁棒性）

数据的噪音——正则项（Tikhonov正则）

丢弃法：在层之间加入噪音（层之间，随机噪音）



如何丢弃？

对x加入噪音得到x'，我们希望E(x')=x（期望值不变）

丢弃法就是对每个元素做如下扰动：
$$
x'_i=\begin{cases}0\ \ \ \ \ \  \text{with prbability }p\\\frac{x_i}{1-p}\  \ \ \text{otherwise}\end{cases}
$$

（p的概率把输入扔掉，并保持期望不变）



如何实现？

通常将丢弃法作用在隐藏全连接层的输出上，如：
$$
\mathbf{h}=\sigma(\mathbf{W_1x+b_1})\\\mathbf{h'}=\text{dropout}(\mathbf{h})\\\mathbf{o}=\mathbf{W_2h'+b_2}\\\mathbf{y}=\text{softmax}(\mathbf{o})
$$
![image-20250324171347655](C:\Users\SmaI\AppData\Roaming\Typora\typora-user-images\image-20250324171347655.png)

（丢弃法等正则项只在训练中使用，不在预测中使用）

 Summary：

- 丢弃法将一些输出项随机置0来控制模型复杂度
- 常作用于多层感知机的隐藏输出上
- 丢弃概率是控制模型复杂度的超参数
- 