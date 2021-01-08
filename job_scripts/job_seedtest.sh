#!/bin/bash

#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

#SBATCH --mail-type=END
#SBATCH --mail-user=pieterbouwman98@gmail.com

# module purge
# module load eb
# module load pre2020
# module load Python/3.7.9
# module load torch
# module load cuDNN/7.0.5-CUDA-9.0.176
# module load NCCL/2.0.5-CUDA-9.0.176
# module load nltk
# export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# source /home/lgpu0136/.bashrc
source activate fact
export PYTHONPATH=$PYTHONPATH:$HOME/project
python -m spacy download en


cd $HOME/project/Transparency
output_dir=$HOME/test_seeds

 
python train_and_run_experiments_bc.py --dataset sst --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset imdb --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset yelp --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset amazon --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset 20News_sports --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset tweet --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset snli --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset qqp --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset babi_1 --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset babi_2 --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset babi_3 --encoder vanilla_lstm --data_dir . --output_dir $output_dir --seed 0



 
python train_and_run_experiments_bc.py --dataset sst --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset imdb --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset yelp --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset amazon --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset 20News_sports --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_bc.py --dataset tweet --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset snli --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset qqp --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset babi_1 --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset babi_2 --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0

python train_and_run_experiments_qa.py --dataset babi_3 --encoder ortho_lstm --data_dir . --output_dir $output_dir --seed 0



 
python train_and_run_experiments_bc.py --dataset sst --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_bc.py --dataset imdb --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_bc.py --dataset yelp --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_bc.py --dataset amazon --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_bc.py --dataset 20News_sports --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_bc.py --dataset tweet --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_qa.py --dataset snli --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.1

python train_and_run_experiments_qa.py --dataset qqp --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_qa.py --dataset babi_1 --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_qa.py --dataset babi_2 --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5

python train_and_run_experiments_qa.py --dataset babi_3 --encoder diversity_lstm --data_dir . --output_dir $output_dir --seed 0 --diversity 0.5