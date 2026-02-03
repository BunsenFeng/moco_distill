import os
import json

if __name__ == "__main__":
    files = os.listdir("model_collaboration/data")
    for file in files:
        if file.endswith(".json"):
            data_config_path = os.path.join("model_collaboration/data", file)
            with open(data_config_path, "r") as f:
                data_config = json.load(f)
            data_config["test"] = data_config["dev"]
            with open(os.path.join("model_collaboration/data", file.replace(".json", "_devonly.json")), "w") as f:
                json.dump(data_config, f, indent=4)