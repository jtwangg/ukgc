#!/bin/bash

#SBATCH --job-name=prompt_tuning    # 作业在调度系统中的作业名为myFirstJob
#SBATCH --partition=qigl     # 作业提交的指定队列/分区为normal_test
#SBATCH --nodes=1                   # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=1         # 每节点任务数，GPU任务不需要修改
#SBATCH --cpus-per-task=20           # V100一张卡默认配置3个CPU核心，gpuB一张卡默认配置12个CPU核心,MIG资源一张卡默认配置6个CPU核心(根据卡数自行调整)
#SBATCH --gres=gpu:2               # 申请一块GPU卡
#SBATCH -o slurm_log/%J.out                   # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号
#SBATCH -e slurm_log/%J.err                   # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号
#SBATCH --mail-type=BEGIN,END,FAIL  # 任务开始，结束，失败时邮件通知
#SBATCH --mail-user=jtwang@seu.edu.cn # 邮件通知邮箱





module load anaconda3               # 加载相关依赖
module load cuda-11.8
source activate g_retriever                 # 如果已经在命令行中激活对应环境，提交脚本时需注释此行，推荐保留此行在base环境下提交任务

wandb offline


export WANDB_API_KEY='a8c8c5c8cec45afbb7ad0c32de74589e03155326'



# prompt tuning with question-only
# echo "prompt tuning with question-only"
# python3 train.py --dataset cn15k_baseline --model_name pt_llm --num_epochs 3 --batch_size 16 --max_txt_len 0 --max_memory 80,80


# prompt tuning with random sampled 50 triples
echo "prompt tuning with random sampled 50 triples"
python3 train.py --dataset cn15k_baseline --model_name pt_llm --num_epochs 5 --batch_size 2 --max_txt_len 2096 --max_memory 80,80


# prompt tuning with pcst retrieval
echo "prompt tuning with pcst retrieval"
python3 train.py --dataset cn15k --model_name pt_llm --num_epochs 5 --max_txt_len 2096 --batch_size 2 --max_memory 80,80

