# EmptyDetection
🚀 `ResNet18` 小项目：**识别空车位**

> 依赖



🤔背景 ： 露天停车场一个较好的优势是可以商用遥感卫星，虽然对于大部分普通停车场还不具备这一条件，但是实际应用场景可以对于该模块进行进一步修改，为了更好地简化问题，我们利用遥感卫星拍摄的图像拍摄空车位

> 数据集

本文数据集来自  `https://aistudio.baidu.com/datasetdetail/122491`

大致如下🌈: ❗空车位( **左** ) ⭕有车车位( **右** ) 

<div align="center">
    <img src="./pic/0.jpg" style="display:inline-block; margin: 0 auto; width: 300px;height: 300px;padding:40px">
    <img src="./pic/1.jpg" style="display:inline-block; margin: 0 auto; width: 300px;height: 300px;
    padding:40px">
</div>

* 总样本分布情况：⭕$829$ 个**正类样本** ⭕$258$ 个**负类样本**

* 实际训练时，将图像转为😶‍🌫️ $32\times 32$ 格式然后投入训练

* 切分训练集，测试集从 **整个数据集** 按照 $8:2$ 比例随机切分✔️

> 模型

本项目采用 ResNet18 架构，其结构如下：

<img src="pic/ResNet18.png" style="height: 1000px;">

💫这样的架构足以解决大多数图像分类情况

> 训练

训练了 😊10 个  $\rm epoch$ 无论在训练集，还是验证集都已经达到接近 $99\%$ 的 $\rm accuracy$ ，于是我们将训练好的模型存储在 🟦`Trained` 文件夹

> 推理

运行 `inference.py` 即可使用模型对一个图片进行判断是否为空车位，例如你运行这一 `python` 文件，即可得到模型对于：



<img src = "pic/spot65.jpg" style="width:300px;">

这实际上是有点算比较难以分出的样本，因为边角是有车的，不过我们的模型输出了正确的值 $\rm 0$

