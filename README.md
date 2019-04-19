# GraphNAS-simple

#### Description
GraphNAS-simple is simple version of GraphNAS, and this project Provides the necessary components for GraphNAS to run on
citation dataset.

#### Installation
Ensure that at least PyTorch 1.0.0 is installed. Then run:
>  pip install -r requirements.txt

If you want to run in docker, you can run:
>  docker build -t graphnas -f DockerFile . \
>  docker run -it -v $(pwd):/GraphNAS graphnas python main.py --dataset cora

#### Software Architecture
* |--main.py Program entry, contains the parameters required by the program
* |--trainer.py 训练器，管理GraphNAS的整个搜索过程，主要管理控制器的训练
* |--models
* &nbsp;&nbsp; |--  gnn.py 利用模型描述生成GNN网络
* &nbsp;&nbsp; |--  gnn_citation_manager.py 控制GNN模型的训练过程
* &nbsp;&nbsp; |--  gnn_controller.py 生成GNN网络的描述
* &nbsp;&nbsp; |--  operators.py 当前定义的GNN网络中的算子
* |--eval

#### Instructions

1. xxxx
2. xxxx
3. xxxx

#### Contribution

1. Fork the repository
2. Create Feat_xxx branch
3. Commit your code
4. Create Pull Request


#### Gitee Feature

1. You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2. Gitee blog [blog.gitee.com](https://blog.gitee.com)
3. Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4. The most valuable open source project [GVP](https://gitee.com/gvp)
5. The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6. The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)