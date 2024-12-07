# full finetuning with FSDP
# torchrun --nnodes 1 --nproc_per_node 8  /home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/finetuning.py \
#          --enable_fsdp --lr 1e-5  --num_epochs 1 --batch_size_training 2 \
#          --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
#          --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  \
#          --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" \
#          --custom_dataset.file "/home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/datasets/ocrvqa_dataset.py"  \
#          --run_validation True --batching_strategy padding

# # LoRA finetuning with FSDP
 CUDA_VISIBLE_DEVICES=6,7 \
 torchrun --nnodes 1 --nproc_per_node 2  /home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/finetuning.py \
          --enable_fsdp --use_peft --peft_method lora --num_epochs 1 --batch_size_training 2 \
          --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
          --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  \
          --output_dir /home/work/hdd_data/VLA/training/vla_recipes/outputs/ocrvqa \
          --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" \
          --custom_dataset.file "/home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/datasets/ocrvqa_dataset.py"  \
          --run_validation True --batching_strategy padding  --use_peft  --lr 1e-5

#CUDA_VISIBLE_DEVICES=6,7 \
#torchrun --nnodes 1 --nproc_per_node 1  /home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/finetuning.py \
#         --lr 1e-5  --num_epochs 1 --batch_size_training 2 \
#         --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
#         --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  \
#         --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" \
#         --custom_dataset.file "/home/work/hdd_data/VLA/training/vla_recipes/recipes/quickstart/finetuning/datasets/ocrvqa_dataset.py"  \
#         --run_validation True --batching_strategy padding  --use_peft --peft_method lora

