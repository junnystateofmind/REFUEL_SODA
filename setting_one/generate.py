from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import argparse
import os
import pickle
import random
import time
import subprocess
import numpy as np


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)

def change_format(row):
    new_trajectory = []
    
    # narrative를 trajectory의 첫 발화로 포함
    if "narrative" in row and row["narrative"]:  # narrative가 존재할 경우 추가
        new_trajectory.append({
            "role": "narrative",  # narrative는 시스템이 제공하는 설명으로 설정
            "content": row["narrative"]
        })
    
    # 대화 발화 순서에 따라 role을 user와 assistant로 교대 지정
    for i, utterance in enumerate(row["dialogue"]):
        role = "user" if i % 2 == 0 else "assistant"  # 짝수는 user, 홀수는 assistant
        new_trajectory.append({'role': role, 'content': utterance})  # 필드 이름 변경
    
    # 새로운 trajectory를 row에 추가
    row["trajectory"] = new_trajectory
    return row


# 중복 제거 함수
unique_traj = {}
def check_redundant(tokenizer, row):
    dialogue_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in row["trajectory"]])
    tokenized = tokenizer(dialogue_text, truncation=True, padding=True, return_tensors="pt").input_ids.tolist()
    tokenized = tuple(tokenized[0])
    if tokenized in unique_traj:
        return False
    unique_traj[tokenized] = 1
    return True


def call_scripts(args, seed, gen_type):
    if gen_type == 'response':
        try:
            subprocess.run(['python', './setting_one/response_generator.py', \
                            '--dataset', f'{os.path.join(args.output_dir, "temp.pkl")}', \
                            '--temperature', f'{args.temperature}', \
                            '--maxlen', f'{args.maxlen}', \
                            '--world_size', f'{args.world_size}', \
                            '--model', f'{args.model}', \
                            '--seed', f'{seed}', \
                            '--num_turns', f'{args.num_turns}', \
                            '--dtype', f'{args.dtype}'], check=True)
        except:
            return False
    else:
        try:
            subprocess.run(['python', './setting_one/user_generator.py', \
                            '--dataset', f'{os.path.join(args.output_dir, "temp.pkl")}', \
                            '--temperature', f'{args.temperature}', \
                            '--maxlen', f'{args.maxlen}', \
                            '--world_size', f'{args.world_size}', \
                            '--model', f'{args.user_model}', \
                            '--seed', f'{seed}', \
                            '--num_turns', f'{args.num_turns}', \
                            '--dtype', f'{args.dtype}'], check=True)
        except:
            return False
    return True


def call_scripts_wrapper(args, seed, gen_type):
    while not call_scripts(args, seed, gen_type):
        time.sleep(20)
        print(f'error when generating {gen_type}')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--user_model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")

    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_repo", type=str, default="")

    parser.add_argument("--dataset", type=str, default="allenai/soda")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument("--num_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--num_data", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":

    # 초기화
    args = parse_arguments()
    set_seed(args.seed)
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 전처리
    if args.num_data != 0:
        dataset = dataset.select(range(args.num_data))
    dataset = dataset.map(change_format)
    dataset = dataset.filter(lambda row: check_redundant(tokenizer, row))

    # 초기 프롬프트 저장
    trajectory = []
    for i in range(len(dataset)):
        # narrative를 포함한 첫 발화만 저장
        if dataset[i]['trajectory']:
            trajectory.append({"trajectory": dataset[i]['trajectory']})  # 딕셔너리로 저장
    
    # Save initial prompt
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Initial prompt saved to {os.path.join(args.output_dir, "temp.pkl")}')

    # temp.pkl 내용 일부 출력
    print("\n[DEBUG] Initial temp.pkl content:")
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
        temp_data = pickle.load(handle)
        print(temp_data[:2])  # 처음 두 개만 출력 (필요 시 조정)

    # Generate and expand for num_turns
    for turn in range(args.num_turns):
        # 응답 생성
        call_scripts_wrapper(args, args.seed, gen_type='response')

        # temp.pkl 내용 읽기
        with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
            temp_data = pickle.load(handle)
        print(f"\n[DEBUG] temp.pkl content after turn {turn}:")
        print(temp_data[:2])  # 처음 두 개만 출력 (필요 시 조정)

        # 업데이트된 데이터를 다시 저장
        with open(os.path.join(args.output_dir, 'temp.pkl'), 'wb') as handle:
            pickle.dump(temp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # temp.pkl의 traj 데이터만 추출해서 허깅페이스에 업로드
        temp_dataset = Dataset.from_dict({"trajectory": [entry["trajectory"] for entry in temp_data]})
        temp_dataset.push_to_hub(args.output_repo + f'_turn_{turn}_ckp')
        print(f'Uploaded turn {turn} checkpoint to Hugging Face Hub.')
        
        if i < args.num_turns - 1:
            call_scripts_wrapper(args, args.seed + 20000, gen_type='user')

    # 최종 데이터 저장
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
        final_trajectory = pickle.load(handle)

    # traj만 사용해서 최종 데이터 업로드
    final_dataset = Dataset.from_dict({"trajectory": [entry["trajectory"] for entry in final_trajectory]})
    final_dataset.push_to_hub(args.output_repo)
    print(f'Final data uploaded to Hugging Face Hub: {args.output_repo}')

