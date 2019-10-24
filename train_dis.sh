#!/bin/bash

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"

cd $THIS_DIR
pip3 install /mnt/cephfs_new_wj/vc/yudongdong/mmdet/files/python3env/torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl
pip3 install mpi4py
pip3 install lmdb
ENV_FILE=test_env.sh
source /mnt/cephfs_new_wj/vc/yudongdong/mmdet/files/${ENV_FILE}
cat /mnt/cephfs_new_wj/vc/yudongdong/mmdet/files/bash_env.sh >> ~/.bash_profile
source ~/.bash_profile
sudo apt-get install unzip

cd /
cp -r /mnt/cephfs_new_wj/vc/yuankun/data ./

echo "Training: "
PYTHON=${PYTHON:-"python3"}
NNODES=$((${ARNOLD_SERVER_NUM:-0} + ${ARNOLD_WORKER_NUM:-1}))
RANK=${ARNOLD_ID:-0}
MASTER_ADDR=${METIS_SERVER_0_HOST:-${METIS_WORKER_0_HOST:-"127.0.0.1"}}
MASTER_PORT=${METIS_SERVER_0_PORT:-${METIS_WORKER_0_PORT:-'29500'}}
echo $PYTHON, $NNODES, $RANK, $MASTER_ADDR, $MASTER_PORT

cd $THIS_DIR

MPIRUN python3 -m torch.distributed.launch 8 train.py ./configs/MsrConfig.yml
