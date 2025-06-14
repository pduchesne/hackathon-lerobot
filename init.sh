curl -L -o install_lerobot_hf_space.sh   https://huggingface.co/datasets/pepijn223/lerobot-install/raw/main/install_lerobot_hf_space.sh
chmod +x install_lerobot_hf_space.sh

./install_lerobot_hf_space.sh

# reload env
source ~/.bashrc

conda activate lerobot

# if you want to redirect logs to wandb
wandb login

huggingface-cli login

cd lerobot/

pip install transformers num2words accelerate

python lerobot/scripts/train.py   --dataset.repo_id=eschnou/lerobot-hackaton-red5   --policy.path=lerobot/smolvla_base   --output_dir=outputs/train/red5_01   --job_name=red5_01   --policy.device=cuda   --wandb.enable=true --steps=1000