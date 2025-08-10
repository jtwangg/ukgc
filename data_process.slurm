#!/bin/bash

#SBATCH --job-name=src.dataset.preprocess    # 作业在调度系统中的作业名为myFirstJob
#SBATCH --partition=qigl     # 作业提交的指定队列/分区为normal_test
#SBATCH --nodes=1                   # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=1         # 每节点任务数，GPU任务不需要修改
#SBATCH --cpus-per-task=12           # V100一张卡默认配置3个CPU核心，gpuB一张卡默认配置12个CPU核心,MIG资源一张卡默认配置6个CPU核心(根据卡数自行调整)
#SBATCH --gres=gpu:1                # 申请一块GPU卡
#SBATCH -o slurm_log/%J.out                   # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号
#SBATCH -e slurm_log/%J.err                   # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号
#SBATCH --mail-type=BEGIN,END,FAIL  # 任务开始，结束，失败时邮件通知
#SBATCH --mail-user=jtwang@seu.edu.cn # 邮件通知邮箱





module load anaconda3               # 加载相关依赖
module load cuda-11.8
source activate g_retriever                 # 如果已经在命令行中激活对应环境，提交脚本时需注释此行，推荐保留此行在base环境下提交任务

wandb offline




# echo "cn15k preprocess"
# python3 -m src.dataset.preprocess.cn15k_train
# echo "cn15k process retrival by pcst"
# python3 -m src.dataset.cn15k_train



# echo "nl27k preprocess"
# python3 -m src.dataset.preprocess.nl27k_train
# echo "nl27k process retrival by pcst"
# python3 -m src.dataset.nl27k_train



echo "ppi5k preprocess"
python3 -m src.dataset.preprocess.ppi5k_train
echo "ppi5k process retrival by pcst"
python3 -m src.dataset.ppi5k_train