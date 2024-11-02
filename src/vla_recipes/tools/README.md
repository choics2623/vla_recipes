## Hugging Face llama 가중치를 공식 Llama형식으로 변환함.
transformer 패키지의 convert_llama_weights_to_hf.py 스크립트에 대한 역변환.

## 0단계: Llama 형식으로 변환
- 변환된 가중치를 위한 출력 디렉토리(예: test70B)를 생성함.  
- 공식 llama 다운로드에서 params.json 파일을 해당 디렉토리로 복사함.  
- 변환 스크립트를 실행함. model-path는 Hugging Face 허브 모델 또는 로컬 hf 모델 디렉토리가 될 수 있음.
```
python -m vla_recipes.tools.convert_hf_weights_to_llama --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --output-dir test70B --model-size 70B
```



## 1단계: 추론  
공식 llama 3 추론 코드를 통해 잘 변환 됐는지 검증 (아직 미구현)
```
torchrun --nproc_per_node 8 example_chat_completion.py --ckpt_dir ./test70B --tokenizer_path ${llama_3_dir}/tokenizer.model
```

검증을 위해, 변환된 가중치를 공식 llama 2 가중치와 비교함  
```
python compare_llama_weights.py test70B ${Llama-3-70B-Instruct_dir}
```