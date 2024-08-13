# Prepare the environment
```shell
git clone  https://github.com/dvlab-research/LISA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
# Replace files
```
1. Replace ./model/LISA.py to our LISA.py
2. Replace ./model/llava/model/language_model/llava_llama.py to our llava_llama.py
3. Add  our my_modeling_llama.py to ./model/llava/model/language_model/
```
# Inference
```shell
python chat.py --version='xinlai/LISA-13B-llama2-v1'
```
After that, input the text prompt and then the image path. For example，
```
- Please input your prompt: Where can the driver see the car speed in this image? Please output segmentation mask.
- Please input the image path: imgs/example1.jpg
```