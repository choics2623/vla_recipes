Hugging Face의 llama 가중치를 공식 llama consolidated 형식으로 변환하는 코드
transformer 패키지의 convert_llama_weights_to_hf.py 스크립트의 역변환하는 코드.

Step 0: Convert to consolidated format  
---
- 변환된 가중치를 위한 출력 디렉토리를 생성. e.g test70B와 같이 생성.
- 공식 llama 다운로드에서 params.json 파일을 해당 디렉토리에 복사합.
- 변환 스크립트를 실행. model-path는 Hugging Face 허브 모델이거나 로컬 hf 모델 디렉토리일 수 있음.
```
python -m llama_recipes.tools.convert_hf_weights_to_llama --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --output-dir test70B --model-size 70B
```
Step 1: Run inference
---
- llama 3 inference 코드를 통해 확인. chat 혹은 text completion을 통해 확인
```
torchrun --nproc_per_node 8 example_chat_completion.py --ckpt_dir ./test70B --tokenizer_path ${llama_3_dir}/tokenizer.model
```

Step 2: 검증을 위해, 변환된 가중치를 공식 llama 2 가중치와 비교.
---
```
python compare_llama_weights.py test70B ${Llama-3-70B-Instruct_dir}
```