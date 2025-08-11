#!/bin/bash
#SBATCH -J mixtral_infer           # 作业名
#SBATCH -p gpu                     # 分区
#SBATCH --gres=gpu:2               # 2 张 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 02:00:00                # 预计 2 小时
#SBATCH -o %x-%j.out               # 输出日志

##############################################################################
# 0. 先确保本机已安装 InfiniCore runtime 库到 $HOME/.infini
#    若你刚刚 build 完但还没安装，请先在登录节点执行
#    -----------------------------------------------------------
#    cd ~/InfiniCore    && xmake install
#    cd ~/InfiniCore-Infer && xmake install   # 如果 Infer 项目也带 install
##############################################################################

######################## 1. 依赖路径 #########################################
# InfiniCore (绝对路径)
export INFINI_ROOT=/home/hot_wind/.infini
# 将所有动态库目录放到最前，确保优先加载自编译版本
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"

# 如果你没有执行 xmake install，而是直接用 build 目录中的 .so，
# 请把下面这行取消注释，并把路径替换成实际 build 路径
# export LD_LIBRARY_PATH=$HOME/InfiniCore/build/linux/x86_64/release:$LD_LIBRARY_PATH

# NCCL —— 使用与 InfiniCore 相同目录下的 libnccl.so
# 如果 ~/.infini/lib 中包含自编译的 libnccl.so，则直接复用
export NCCL_HOME=$INFINI_ROOT/lib
# 追加（与上方路径相同，但便于后续单独引用 NCCL_HOME）
export LD_LIBRARY_PATH="$NCCL_HOME:$LD_LIBRARY_PATH"

# CUDA 工具链（compute-sanitizer 等）
# 集群通常在 module 里设置好；若无 module，可手动指定
#module load cuda/12.2            # 示例；按实际版本修改
# 或者手动:
# export CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

######################## 2. 可选调试环境变量 #################################
# 显示 NCCL 详细日志
export NCCL_DEBUG=WARN
# 避免 IB/网卡特性缺失时报错，可按需开启
# export NCCL_P2P_DISABLE=1
# export NCCL_SHARP_DISABLE=1

######################## 3. 启动推理 #########################################
# PYTHONPATH 内如果需要引用本地源码，可以再加:
# export PYTHONPATH=$HOME/InfiniCore-Infer:$PYTHONPATH

# ■★★★  最终执行命令  ★★★■
srun --gpus=2 --cpus-per-task=16 --mem=256G --export=ALL compute-sanitizer python scripts/mixtral.py --nvidia /home/shared/models/Mixtral-8x7B-v0.1-HF/ 2
##############################################################################