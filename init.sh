
# Use script from pepijn223 to bootstrap most of the env
curl -L -o install_lerobot_hf_space.sh   https://huggingface.co/datasets/pepijn223/lerobot-install/raw/main/install_lerobot_hf_space.sh
chmod +x install_lerobot_hf_space.sh
./install_lerobot_hf_space.sh

# reload env after bootstrap
source ~/.bashrc

# activate the python env
conda activate lerobot

# install modules required for the smolVLA model
pip install transformers num2words accelerate

# this is sometimes required to let huggingface-cli link with git auth
git config --global credential.helper store

# Log to HF to enable pull/pull of models from HF repos
# Have your HF token at hand
huggingface-cli login

# if you want to redirect logs to wandb
#wandb login

cd lerobot/

# be sure to replace at least the repo_id
python lerobot/scripts/train.py   --dataset.repo_id=eschnou/lerobot-hackaton-red5   --policy.path=lerobot/smolvla_base   --output_dir=outputs/train/red5_01   --job_name=red5_01   --policy.device=cuda   --wandb.enable=true --steps=1000

# push the trained model to the repo
#python lerobot/scripts/push_pretrained.py --pretrained_path=outputs/train/red5_01/checkpoints/last/pretrained_model --repo_id=eschnou/lerobot-hackaton-red5