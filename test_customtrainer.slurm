#!/bin/bash

#SBATCH --job-name=test_customtrainer    # 作业在调度系统中的作业名为myFirstJob
#SBATCH --partition=qigl     # 作业提交的指定队列/分区为normal_test
#SBATCH --nodes=1                   # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=1         # 每节点任务数，GPU任务不需要修改
#SBATCH --cpus-per-task=20           # V100一张卡默认配置3个CPU核心，gpuB一张卡默认配置12个CPU核心,MIG资源一张卡默认配置6个CPU核心(根据卡数自行调整)
#SBATCH --gres=gpu:2                # 申请一块GPU卡
#SBATCH -o slurm_log/%J.out                   # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号
#SBATCH -e slurm_log/%J.err                   # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号
#SBATCH --mail-type=BEGIN,END,FAIL  # 任务开始，结束，失败时邮件通知
#SBATCH --mail-user=jtwang@seu.edu.cn # 邮件通知邮箱





module load anaconda3               # 加载相关依赖
module load cuda-11.8
source activate g_retriever                 # 如果已经在命令行中激活对应环境，提交脚本时需注释此行，推荐保留此行在base环境下提交任务

wandb offline


export WANDB_API_KEY='a8c8c5c8cec45afbb7ad0c32de74589e03155326'
export WANDB_PROJECT="graph_prompt_tuning"
export PYTHONUNBUFFERED=1





# echo "nl27k_tc, llm_frozen=False, zero2_nocpu, mbs=8 "
# stdbuf -o0 -e0 python3 -u test_ds_customtrainer.py --dataset nl27k_tc --model_name graph_llm_ds_customtrainer --exp_name loss7 --num_epochs 1 --batch_size 32 --micro_batch_size 8 --max_txt_len 1500 --llm_frozen True --deepspeed "ds_config_zero2_nocpu.json" --eval_batch_size 8 --max_new_tokens 10
# echo "nl27k_tc, llm_frozen=False, zero2_nocpu, mbs=8 "
# stdbuf -o0 -e0 python3 -u test_ds_customtrainer.py --dataset nl27k_tc --model_name graph_llm_ds_customtrainer --exp_name loss7 --num_epochs 1 --batch_size 32 --micro_batch_size 8 --max_txt_len 1500 --llm_frozen False --deepspeed "ds_config_zero2_nocpu.json" --eval_batch_size 8 --max_new_tokens 10



echo "nl27k_tc, llm_frozen=True, zero2_nocpu, mbs=8 "
deepspeed --num_gpus=2 --master_port 29520 test_ds_customtrainer.py --dataset nl27k_tc --model_name graph_llm_ds_customtrainer --exp_name loss7 --num_epochs 1 --batch_size 32 --micro_batch_size 8 --max_txt_len 1500 --llm_frozen True --deepspeed "ds_config_zero2_nocpu.json" --eval_batch_size 8 --max_new_tokens 10
echo "nl27k_tc, llm_frozen=False, zero2_nocpu, mbs=8 "
deepspeed --num_gpus=2 --master_port 29521 test_ds_customtrainer.py --dataset nl27k_tc --model_name graph_llm_ds_customtrainer --exp_name loss7 --num_epochs 1 --batch_size 32 --micro_batch_size 8 --max_txt_len 1500 --llm_frozen False --deepspeed "ds_config_zero2_nocpu.json" --eval_batch_size 8 --max_new_tokens 10
