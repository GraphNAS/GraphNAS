# GraphNAS-simple

GraphNAS-simple 是 GraphNAS的简化版，提供了GraphNAS运行的必备组件
##搭建运行环境
* 系统环境
> pip install -r requirements.txt \
> pip install pytorch 
* docker
> docker build -t graphnas -f DockerFile . \
> docker run -it -v $(pwd):/GraphNAS graphnas python main.py




[ood]:**pytorch是GraphNAS的依赖库，没有将其加入requirements.txt,是因为pytorch的包较大，安装过程中容易下载出错，导致docker创建失败**

##程序说明
* |--main.py 程序入口，包含了程序需要的参数
* |--trainer.py 训练器，管理GraphNAS的整个搜索过程，主要管理控制器的训练
* |--models
* &nbsp; |--  gnn.py 利用模型描述生成GNN网络
* &nbsp; |--  gnn_citation_manager.py 控制GNN模型的训练过程
* &nbsp; |--  gnn_controller.py 生成GNN网络的描述
* &nbsp; |--  operators.py 当前定义的GNN网络中的算子
* |--eval
=======

#### 软件架构
软件架构说明





