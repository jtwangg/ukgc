#!/bin/bash

#SBATCH --job-name=test    # 作业在调度系统中的作业名为myFirstJob
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




# 测试链接预测任务，graph中去掉包含tail实体的


# prompt tuning方法
# echo "nl27k, prompt tuning with random sampled 50 triples"
# python3 test.py --dataset nl27k_baseline --model_name pt_llm --num_epochs 5 --max_txt_len 2096

# echo "prompt tuning with pcst retrieval"
# python3 train.py --dataset nl27k --model_name pt_llm --num_epochs 5 --max_txt_len 2096



# graph prompt tuning 方法
# echo "nl27k, graph prompt tuning with random sampled 50 triples"
# python3 test.py --dataset nl27k_baseline --model_name graph_llm --num_epochs 3 --max_txt_len 2096

# echo "cn15k G-Retriever"
# python3 test.py --dataset nl27k --model_name graph_llm --num_epochs 3 --max_txt_len 2096





# echo "ppi5k_baseline_cp, pt_llm, random sampled 50 triples, llm_frozen=false, fp16=False, zero2, mbs=8"
# stdbuf -o0 -e0 python3 -u test_ds.py --dataset ppi5k_baseline_cp --model_name pt_llm --num_epochs 3 --batch_size 16 --micro_batch_size 8 --max_txt_len 2000 --llm_frozen False --deepspeed "ds_config_zero2_nocpu.json"




# echo "nl27k, random sampled 50 triples, graph_llm_ds, llm_frozen=false, fp16, zero2, mbs=8"
# stdbuf -o0 -e0 python3 -u test_ds.py --dataset nl27k_baseline_cp --model_name graph_llm_ds --num_epochs 3 --batch_size 32 --micro_batch_size 8 --max_txt_len 1700 --llm_frozen False --eval_batch_size 8

echo "nl27k_baseline_cp, llm_ds, llm_frozen=false, zero2"
stdbuf -o0 -e0 python3 -u test_ds.py --dataset nl27k_baseline_cp --model_name llm_ds --num_epochs 3 --batch_size 64 --micro_batch_size 8 --max_txt_len 1800 --llm_frozen False --deepspeed "ds_config_zero2_nocpu.json" --eval_batch_size 8
echo "nl27k_cp, llm_ds, llm_frozen=false, zero2"
stdbuf -o0 -e0 python3 -u test_ds.py --dataset nl27k_cp --model_name llm_ds --num_epochs 3 --batch_size 64 --micro_batch_size 16 --max_txt_len 1800 --llm_frozen False --deepspeed "ds_config_zero2_nocpu.json" --eval_batch_size 8
