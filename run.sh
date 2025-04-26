
#torchrun --nproc_per_node=2  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10

export WANDB_MODE=disabled
export WANDB_PROJECT=diloco
export WANDB_ENTITY=eb-lab
export WANDB_RUN_GROUP=jax-14m

CUDA_VISIBLE_DEVICES=6,7 python jax_diloco.py   