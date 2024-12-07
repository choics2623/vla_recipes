# CUDA_VISIBLE_DEVICES=6,7 \
# torchrun --nnodes 1 --nproc_per_node 2 --master_port 29501 /home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/finetuning.py \
#         --enable_fsdp --use_peft --peft_method lora --num_epochs 1 --batch_size_training 2 \
#         --model_name meta-llama/Llama-3.2-3B-Instruct \
#         --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  \
#         --output_dir /home/work/hdd_data/VLA/training/vla_recipes/outputs/bllossom_datset \
#         --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" \
#         --custom_dataset.file "/home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/datasets/sharegpt.py"  \
#         --run_validation True --batching_strategy padding  --use_peft  --lr 1e-5


CUDA_VISIBLE_DEVICES=6,7 \
torchrun --nnodes 1 --nproc_per_node 1 --master_port 29501 /home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/finetuning.py \
        --use_peft --peft_method lora --quantization 4bit --num_epochs 1 --batch_size_training 2 \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  \
        --output_dir /home/work/hdd_data/VLA/training/vla_recipes/outputs/bllossom_datset \
        --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" \
        --custom_dataset.file "/home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/datasets/sharegpt.py"  \
        --run_validation True --batching_strategy padding --lr 1e-5