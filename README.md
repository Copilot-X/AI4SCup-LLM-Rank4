# 1、基本环境
    Python版本： Python 3.10.12
    cuda版本： 12.2
    cudnn： 8.3.1
    显卡： RTX-3090-24G
    安装依赖包：
        pip install -r requirements.txt

# 2、训练脚本
    ① 修改gemma-7b模型路径
        进入train.sh, 修改MODEL的真实模型路径值

    ② 启动训练脚本
        sh train.sh

# 3、比赛链接：https://bohrium.dp.tech/competitions/3793785610?tab=introduce