import os
import sys
import json
import torch
import shutil
import argparse
import importlib
from model_collaboration.data import eval
from multiprocessing import Pool
from model_collaboration.utils import distributed_sft
from model_collaboration.method import distributed_generation

if __name__ == "__main__":
    
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    method_name = config.get("method")
    task = config["task"]
    task_type = config["task_type"]
    gpu_ids = config["gpu_ids"]
    model_names = config["model_names"]
    hyperparameters = config["hyperparameters"]
    rounds = config.get("rounds", 3)

    run_path = f"model_collaboration/moco_distill/logs/{task}_{len(model_names)}_{method_name}_supervised_kd/"
    if os.path.exists(run_path):
        raise RuntimeError(f"Run path {run_path} already exists. Please remove it before running.")
    os.makedirs(run_path, exist_ok=True)

    distributed_generation.update_generation_hyperparameters(
        hyperparameters.get("max_response_length", 512),
        hyperparameters.get("temperature", 0.7),
        hyperparameters.get("top_p", 0.9),
        hyperparameters.get("batch_size", 32)
    )

    score_dict = {
        "single": [],
        "multi": [],
        "single_str": []
    }
    score_dict_file_name = os.path.join(run_path, "score_dict.json")

    for round_idx in range(rounds+1):
        print(f"=== Round {round_idx} ===")
        if round_idx > 0:

            # run it on the dev set to get SFT data
            print(f"--- Generating SFT data on dev set ---")
            multi_config = {
                "method": method_name,
                "task": task + "_devonly",
                "task_type": task_type,
                "gpu_ids": gpu_ids,
                "model_names": model_names,
                "hyperparameters": hyperparameters
            }

            with open(os.path.join(run_path, "multi_devonly_config.json"), "w") as f:
                json.dump(multi_config, f, indent=4)

            if "swarm_base_path" in hyperparameters:
                # remove the swarm_base_path directory
                shutil.rmtree(hyperparameters["swarm_base_path"], ignore_errors=True)
            
            os.system(f"python main.py -c {os.path.join(run_path, 'multi_devonly_config.json')}")

            # identify the log file
            for file in os.listdir("model_collaboration/logs/"):
                if file.startswith(f"{task}_devonly_{len(model_names)}") and (file.endswith(f"_{method_name}.json") or file.endswith(f"_{method_name.split('_')[-1]}.json")):
                    log_filename = os.path.join("model_collaboration/logs/", file)
                    break
            # move the log file to the run path
            shutil.move(log_filename, os.path.join(run_path, "multi_devonly_" + str(round_idx) + ".json"))

            # prepare the SFT data
            sft_examples = []
            with open(os.path.join(run_path, f"multi_devonly_{round_idx}.json"), "r") as f:
                logs = json.load(f)
            for item in logs["logs"]:
                sft_examples.append({
                    "prompt": item["input"],
                    "completion": item["output"]
                })
            sft_data_path = os.path.join(run_path, f"sft_data.jsonl")
            with open(sft_data_path, "w") as f:
                for example in sft_examples:
                    f.write(json.dumps(example) + "\n")

            # SFT all models on it
            distributed_sft.distributed_sft(
                model_names,
                [sft_data_path] * len(model_names),
                gpu_ids,
                [os.path.join(run_path, f"model_{i}_new") for i in range(len(model_names))],
                epoch=1
            )

            for i in range(len(model_names)):
                if os.path.exists(os.path.join(run_path, f"model_{i}")):
                    shutil.rmtree(os.path.join(run_path, f"model_{i}"))
                # rename model_i_new as model_i
                os.rename(os.path.join(run_path, f"model_{i}_new"), os.path.join(run_path, f"model_{i}"))
            # update model names to the new SFTed models
            model_names = [os.path.join(run_path, f"model_{i}") for i in range(len(model_names))]

        # evaluate the single models
        single_scores = []
        for i in range(len(model_names)):
            print(f"--- Evaluating model {i+1} of {len(model_names)}: {model_names[i]} ---")
            model_name = model_names[i]

            single_config = {
                "method": "single_model",
                "task": task,
                "task_type": task_type,
                "gpu_ids": [gpu_ids[0]],  # use only one GPU for single model evaluation
                "model_names": [model_name],
                "hyperparameters": hyperparameters
            }

            with open(os.path.join(run_path, "single_config.json"), "w") as f:
                json.dump(single_config, f, indent=4)

            os.system(f"python main.py -c {os.path.join(run_path, 'single_config.json')}")

            # identify the file
            simple_model_name = model_name.split("/")[-1]
            for file in os.listdir("model_collaboration/logs/"):
                if file.startswith(f"{task}_{simple_model_name}") and file.endswith("_single_model.json"):
                    log_filename = os.path.join("model_collaboration/logs/", file)
                    break
            with open(log_filename, "r") as f:
                logs = json.load(f)
            avg_score = logs["avg_test_score"]
            single_scores.append(avg_score)
            print(f"Model: {model_name}, single model {task} score: {avg_score}")
            # move the log file to the run path
            shutil.move(log_filename, os.path.join(run_path, "single_" + str(round_idx) + "_" + str(i) + ".json"))
        score_dict["single"].append(single_scores)
        single_avg = sum(single_scores) / len(single_scores) * 100
        single_std = torch.std(torch.tensor(single_scores)).item() * 100
        score_dict["single_str"].append(f"{single_avg:.2f} ({single_std:.2f})")

        with open(score_dict_file_name, "w") as f:
            json.dump(score_dict, f, indent=4)

        # evaluate the model collaboration
        print(f"--- Evaluating model collaboration ---")
        multi_config = {
            "method": method_name,
            "task": task,
            "task_type": task_type,
            "gpu_ids": gpu_ids,
            "model_names": model_names,
            "hyperparameters": hyperparameters
        }

        with open(os.path.join(run_path, "multi_config.json"), "w") as f:
            json.dump(multi_config, f, indent=4)

        if "swarm_base_path" in hyperparameters:
            # remove the swarm_base_path directory
            shutil.rmtree(hyperparameters["swarm_base_path"], ignore_errors=True)

        os.system(f"python main.py -c {os.path.join(run_path, 'multi_config.json')}")

        # identify the file
        for file in os.listdir("model_collaboration/logs/"):
            if file.startswith(f"{task}_{len(model_names)}") and file.endswith(f"_{method_name}.json") or file.endswith(f"_{method_name.split('_')[-1]}.json"):
                log_filename = os.path.join("model_collaboration/logs/", file)
                break
        with open(log_filename, "r") as f:
            logs = json.load(f)
        avg_score = logs["avg_test_score"]
        score_dict["multi"].append(avg_score * 100.0)
        # if round_idx < rounds:
        shutil.move(log_filename, os.path.join(run_path, "multi_" + str(round_idx) + ".json"))

        with open(score_dict_file_name, "w") as f:
            json.dump(score_dict, f, indent=4)

    # move every json file except score_dict.json in run_path to run_path/generation_logs/
    generation_logs_path = os.path.join(run_path, "generation_logs")
    os.makedirs(generation_logs_path, exist_ok=True)
    for file in os.listdir(run_path):
        if file.endswith(".json") and file != "score_dict.json":
            shutil.move(os.path.join(run_path, file), os.path.join(generation_logs_path, file))