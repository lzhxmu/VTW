# Visual Tokens Withdrawal 
Code release for "Boosting Multimodal Large Language Models with Visual Tokens
Withdrawal for Rapid Inference" 


## Experiments Environment
### Set Up the Dependencies as:
```
# install llava
conda create -n vtw python=3.10 -y
conda activate vtw
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# install lmms-eval
cd lmms-evaluation
pip install -e .
```
### Modify a Few Lines of Code
```python 
# 1.Open file  /anaconda3/envs/vtw/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
# 2.Modify code 
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
# to:
cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
# Note that there are a total of 3 identical lines of code that need to be modified
```
## Chatbot
```
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b   \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --use_vtw
``` 

## Search Visual Tokens Withdrawal  Layer K
```bash
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"  \
    --tasks scienceqa_img --batch_size 1 \
    --log_samples_suffix llava-1.5-7b \
    --output_path ./logs/ \
    --limit 20 --findk
```


## Evaluation Baseline
### Command
```bash
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"  \
    --tasks scienceqa_img --batch_size 1 \
    --log_samples_suffix llava_7b \
    --output_path ./logs/7b/ 
```
### You will get
![alt text](./assets/baseline.png)

## Evaluation with Visual Tokens Withdrawal
### Command
```bash
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"  \
    --tasks scienceqa_img --batch_size 1 \
    --log_samples_suffix llava_7b \
    --output_path ./logs/7b/ \
    --use_vtw --k=15    # Use the searched K or specify K manually 
```
### You will get
![alt text](./assets/vtw.png)
