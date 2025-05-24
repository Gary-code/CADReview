import json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import argparse
import base64
import re
import ast
import os
import sys
from collections import defaultdict
import concurrent.futures
# sys.path.append("/vepfs/fs_users/weiyuancheng-test/LLM4IE-code/wyc-go")
import random
from openai import OpenAI   
import pandas as pd
from datetime import datetime


openai_no_support_model = ['yi-vl-6b-chat']

def list_to_jsonl(result_list:list, output_path:str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in result_list:
            f.write(json.dumps(item) + '\n')

def process_item(item, client_pool, model_name="qwen2-vl-7b-instruct", temperature=0, top_p=1, stream="False"):
    n = random.randint(0, len(client_pool) - 1)
    client = client_pool[n]
    image_base64 = []
    for image_path in item["images"]:
        with open(image_path, "rb") as image:
            image_base64.append(base64.b64encode(image.read()).decode("utf-8"))
    input_query = item["messages"][1]["content"].replace("<image>","")
    # input_query += "\nplease output the block id dirctly!"
    # input_query = "Please help me describe the content of the picture"
    # First_round_message = item["messages"][:2]
    First_round_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64[0]}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64[1]}"},
                }
            ]
        }
    ]

    First_round_response = client.chat.completions.create(
        model = model_name if model_name not in openai_no_support_model else 'default-lora',
        messages=First_round_message,
        timeout=1000,
        temperature=temperature,
        top_p=top_p,
        max_tokens=2048,
        stream=stream
    )
    # print(First_round_response)
    item["prediction"] = First_round_response.choices[0].message.content
    return item


def main(task='qg', port_num=8, model_name='llama3_2-11b-vision-instruct', infer_type="sft", step=''):
    device="cuda"
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_data_path',type=str,default=f"./Dataset/feedback_gen/yellow_images/swift_dataset_test.jsonl")
    parser.add_argument('--output_dir',type=str,default=f"./infer_output/{task}")
    parser.add_argument("--additional_information",default=None,type=str,required=False)
    args=parser.parse_args()
    API_POOL = ["9010","9011","9012","9013","9014","9015","9016","9017"]
    API_POOL = API_POOL[:port_num]
    temperature=0.0
    top_p=1.0
    stream = False
    api_key="EMPTY"
    client_pool=[]
    for api_port in API_POOL:
        api_base = f"http://localhost:{api_port}/v1/"
        client = OpenAI(
            base_url = api_base,
            api_key = api_key,
            )
        client_pool.append(client)
        model_name_1 = client.models.list().data[0].id

    data_list = pd.read_json(args.input_data_path,lines=True).to_dict(orient='records')    
    with concurrent.futures.ThreadPoolExecutor(max_workers=180) as executor:

        results = list(tqdm(executor.map(lambda item: process_item(item, client_pool, model_name_1, temperature, top_p, stream), data_list), total=len(data_list)))
    
    if not os.path.exists(args.output_dir):
        print(f"Output directory does not exist. Creating: {args.output_dir}")
        os.makedirs(args.output_dir)
    if args.additional_information is None:
        output_path = os.path.join(args.output_dir, f"pred_{model_name}_{infer_type}_{step}.jsonl")
    else:
        output_path = os.path.join(args.output_dir, f"pred_{model_name}_{infer_type}_{step}.jsonl")
    
    list_to_jsonl(results, output_path)
    print("output_path", output_path)
    
    
    
    
if __name__ == "__main__":
    main(task="feedback_gen", port_num=4, model_name='llama3_2', infer_type="sft_llm", step='3252')
    
    
