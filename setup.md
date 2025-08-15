### just some stuff to
help me set up the mood, set up the franchise hahaha

on lambda :/

start off with the python configs first



then the cloning comes in

```
git clone https://github.com/bendemonium/fused-attention-model.git
```

!!! cd 2 the cloned directory


then set up env

```
python -m venv bb-env
source bb-env/bin/activate
source bb-env/bin/activate
```

```
pip install -r requirements.txt
```

```
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```
```
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

git stuff

```
git config --global user.name "bendemonium"
git config --global user.email "ridhib2422@gmail.com"
git config --global credential.helper store

git remote -v

git add .

sudo apt install git-lfs
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
```

hf login 

```
huggingface-cli login
```

!!! copy .env file onto the root, the export

```
export $(grep -v '^#' .env | xargs)
```

```
# test run

```
pip3 install deepspeed

python -m accelerate.commands.launch scripts/train.py --config configs/mixed_test.yaml --model mixed
```