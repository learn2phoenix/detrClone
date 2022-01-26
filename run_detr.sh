#!/bin/bash
#SBATCH --job-name=detr_training
#SBATCH --time=36:00:00
#SBATCH --qos high
#SBATCH --account abhinav
#SBATCH --cpus-per-task 8
#SBATCH --mem=48gb
#SBATCH --gres gpu:2
#SBATCH --exclude vulcan24,vulcan04,vulcan10,vulcan01
#SBATCH --requeue


source /fs/vulcan-projects/actionbytes/env_file
conda activate detr

EXP_NAME="experiments/q30_e0.4"
RESUME_CMD=""
# Check if any pretrained model exists
if ls "$EXP_NAME"/*.pth 1> /dev/null 2>&1
then
	echo "IN"
	# If already saved a model, pick the latest model
	PRETRAINED_WTS=$(ls -1 "$EXP_NAME"/checkpoint*.pth | sort | tail -n1)
    	RESUME_CMD="--resume $PRETRAINED_WTS"
fi

srun python -m torch.distributed.launch \
--nproc_per_node=2 \
--use_env main.py \
--output_dir $EXP_NAME \
--num_queries 30 \
--eos_coef 0.4 \
--coco_path ../data/pascal/ \
--dataset_file pascal $RESUME_CMD

# experiments/q40_e0.2 - 1000123
# experiments/q50_e0.2 - 1000124
# experiments/q30_e0.3 - 1000125
# experiments/q30_e0.4 - 1000126