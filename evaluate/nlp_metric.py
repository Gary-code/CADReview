import os
import json
import jsonlines
from tqdm import tqdm
from evaluate import load
from my_cider import CHAIR
import json
import argparse
import numpy as np
import pandas as pd
import shutil
BLEU = load("bleu")
ROUGE = load("rouge")
METEOR = load("meteor")
BERTSCORE = load("bertscore")
def find_files(directory, postfix=""):
    output_files = []
    # file_list = os.listdir(directory.replace("stl","clean"))
    for root, dirs, files in os.walk(directory):
        for file in files:
            # name = root.split('/')[-1] + ".scad" 
            if file.endswith(postfix):
            # if file.endswith(".stl") and name in file_list:
                output_files.append(os.path.join(root, file))
    return output_files
def process_file(file_path, data_type=None):

    refs = []
    preds = []

    data = []
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.endswith('.jsonl'):
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                data.append(obj)


    for entry in tqdm(data):
        if data_type is None or entry.get("data_type", "") == data_type:
            if "feedback" in entry.get("gt", ""):
                refs.append([entry.get("gt", "")["feedback"]])
                preds.append(entry.get("pred", "")["feedback"])
    
    if refs:
        results = evaluate_metrics(preds, refs)
        return results
    else:
        return "error, no data eval"

def evaluate_metrics(preds, refs):
    print(len(preds))
    imgids = [i for i in range(len(preds))]
    evaluator = CHAIR(imgids, "")
    cider = evaluator.compute_metric(imgids, preds, refs)
    bleu_score = BLEU.compute(predictions=preds, references=refs)
    rouge = ROUGE.compute(predictions=preds, references=refs)
    meteor = METEOR.compute(predictions=preds, references=refs)
    bertscore = BERTSCORE.compute(predictions=preds, references=refs, lang="en", model_type="microsoft/deberta-large-mnli")
    # bertscore = BERTSCORE.compute(predictions=preds, references=refs, lang="en", model_type="roberta-large")
    # bertscore = BERTSCORE.compute(predictions=preds, references=refs, lang="en", model_type="bert-base-uncased")
    # print(bertscore)
    results = {
        "BLEU-1": bleu_score["precisions"][0],
        "BLEU-2": bleu_score["precisions"][1],
        "BLEU-3": bleu_score["precisions"][2],
        "BLEU-4": bleu_score["precisions"][3],
        "METEOR": meteor['meteor'],
        "ROUGE-L-F": rouge['rougeL'],
        "CIDer": cider,
        "BERTScore": {
            "Precision": float(np.array(bertscore["precision"]).mean()),
            "Recall": float(np.array(bertscore["recall"]).mean()),
            "F1": float(np.array(bertscore["f1"]).mean()),
        }
    }
    return results

def save_to_csv(results, experiment_name, data_type=None):
    csv_path = os.path.join("./evaluate", 'nlp_metric.csv')

    new_row = {
        'experiment': experiment_name,
        'data_type': data_type if data_type else 'all',
        'BLEU-1': results['BLEU-1'],
        'BLEU-2': results['BLEU-2'],
        'BLEU-3': results['BLEU-3'],
        'BLEU-4': results['BLEU-4'],
        'METEOR': results['METEOR'],
        'ROUGE-L-F': results['ROUGE-L-F'],
        'CIDer': results['CIDer'],
        'BERTScore-p': results['BERTScore']['Precision'],
        'BERTScore-r': results['BERTScore']['Recall'],
        'BERTScore-f1': results['BERTScore']['F1'],
    }
    print(new_row)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    

    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="")
    args = parser.parse_args()
        
    json_files = find_files(args.src, postfix=".json")
    json_path = json_files[0]
    

    data_types = set()
    with open(json_path, 'r') as f:
        data = json.load(f)
        for entry in data:
            if "data_type" in entry:
                data_types.add(entry["data_type"])
    

    experiment_name = os.path.basename(json_path).replace('.json', '')

    results_all = process_file(json_path)
    if isinstance(results_all, dict):
        save_to_csv(results_all, experiment_name)
        print(f"所有数据的评估结果: {results_all}")

    for data_type in data_types:
        results = process_file(json_path, data_type)
        if isinstance(results, dict):
            save_to_csv(results, experiment_name, data_type)
            print(f"数据类型 {data_type} 的评估结果: {results}")
    