import json


def save_config(data_config_dict, config_path):
    try:
        with config_path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(data_config_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving config: {e}")


def load_config(config_path):
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def save_hash(hash_string, file_path):
    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write(hash_string)
    except Exception as e:
        print(f"Error saving hash: {e}")


def load_hash(file_path):
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading hash: {e}")
        return None
