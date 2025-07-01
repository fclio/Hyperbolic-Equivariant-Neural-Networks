import json
import os
import csv

def format_dataset_name(name):
    return name.replace("_rot", " - rotation") if "_rot" in name else name

def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def extract_adversarial_results(base_path, dataset):
    attacks = ['fgsm', 'pgd']
    adv_results = {}
    for attack in attacks:
        path = os.path.join(base_path, f"test_{dataset}_{attack}.json")
        data = load_json_if_exists(path)
        if data and "adversarial_attack" in data:
            adv_results[attack.upper()] = {
                epsilon: adv.get("acc@1", "N/A")
                for epsilon, adv in data["adversarial_attack"].items()
            }
    return adv_results


def collect_results(configs, datasets=None, dims=None, model_sizes=None, include_adversarial=False):
    all_rows = []

    # 1. Size experiments (no dim folder)
    if model_sizes is not None:
        for config in configs:
            for dataset in datasets:
                ds_name = format_dataset_name(dataset)
                for size in model_sizes:
                    if size == "normal":
                        folder = f"classification/output/{config}_{dataset}_epoch:200_P4"
                    else:
                        folder = f"classification/output/{config}_{dataset}_epoch:200_P4_{size}"
                    path = os.path.join(folder, "test_results.json")

                    data = load_json_if_exists(path)
                    acc1_val = data.get("acc1_val", "N/A") if data else "N/A"
                    acc1_test = data.get("acc1_test", "N/A") if data else "N/A"

                    # Clean test row
                    all_rows.append([
                        config, ds_name, "N/A", size, acc1_val, acc1_test, "None", "", ""
                    ])

                    if include_adversarial:
                        adv_results = extract_adversarial_results(folder, dataset)
                        if adv_results:
                            for attack, eps_dict in adv_results.items():
                                for eps, acc in eps_dict.items():
                                    all_rows.append([
                                        config, ds_name, "N/A", size, acc1_val, acc1_test, attack, eps, acc
                                    ])

    # 2. Dim experiments (no size folder)
    if dims is not None:
      for config in configs:
        for dataset in datasets:
            ds_name = format_dataset_name(dataset)
            for dim in dims:
                if dim == 512:
                    folder = f"classification/output/{config}_{dataset}_epoch:200_P4"
                else:
                    folder = f"classification/output/{config}_{dataset}_epoch:200_P4_dim:{dim}"
                path = os.path.join(folder, "test_results.json")

                data = load_json_if_exists(path)
                acc1_val = data.get("acc1_val", "N/A") if data else "N/A"
                acc1_test = data.get("acc1_test", "N/A") if data else "N/A"

                # Clean test row
                all_rows.append([
                    config, ds_name, dim, "N/A", acc1_val, acc1_test, "None", "", ""
                ])

                if include_adversarial:
                    adv_results = extract_adversarial_results(folder, dataset)
                    if adv_results:
                        for attack, eps_dict in adv_results.items():
                            for eps, acc in eps_dict.items():
                                all_rows.append([
                                    config, ds_name, dim, "N/A", acc1_val, acc1_test, attack, eps, acc
                                ])

    return all_rows


def collect_results_grouped_separate_rows(configs, datasets=None, dims=None, model_sizes=None):
    grouped_results = {}  # (config, dataset, dim) -> {model_size: (val_acc, test_acc)}

    for config in configs:
        for dataset in datasets:
            ds_name = format_dataset_name(dataset)
            dim_list = dims if dims is not None else [None]
            for dim in dim_list:
                key = (config, ds_name, dim if dim is not None else "N/A")
                grouped_results[key] = {}

                for size in (model_sizes if model_sizes is not None else [None]):
                    if size == "normal" or size is None:
                        if dim is not None:
                            folder = f"classification/output/{config}_{dataset}_epoch:200_P4_dim:{dim}"
                        else:
                            folder = f"classification/output/{config}_{dataset}_epoch:200_P4"
                    else:
                        if dim is not None:
                            folder = f"classification/output/{config}_{dataset}_epoch:200_P4_dim:{dim}_{size}"
                        else:
                            folder = f"classification/output/{config}_{dataset}_epoch:200_P4_{size}"
                    
                    path = os.path.join(folder, "test_results.json")
                    data = load_json_if_exists(path)
                    acc1_val = data.get("acc1_val", "N/A") if data else "N/A"
                    acc1_test = data.get("acc1_test", "N/A") if data else "N/A"

                    grouped_results[key][size if size is not None else "normal"] = (acc1_val, acc1_test)

    return grouped_results
# === CONFIGURATION ===
configs = ["E-CNN", "EQE-CNN", "L-CNN", "LEQE-CNN-3_4"]
datasets = ["Tiny-ImageNet", "CUB-200", "Flower102"]
# datasets = ["Tiny-ImageNet", "CUB-200", "Flower102", "SUN397","PCAM" ]
# dims = [512]
dims = None
model_size =["small","normal","big"]
# model_size = None


# === COLLECT + EXPORT ===
rows = collect_results(configs, datasets, dims,model_size,False)

# Save to CSV
output_file = "classification_results.csv"

if model_size:  # If model_size is not None or empty, write grouped val/test rows
    # Collect grouped results for the new format
    grouped = collect_results_grouped_separate_rows(configs, datasets, dims, model_size)

    header = ["Config", "Dataset", "Dim", "Metric"] + model_size

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for (config, dataset, dim), size_data in grouped.items():
            val_row = [config, dataset, dim, "Val Accuracy"]
            test_row = [config, dataset, dim, "Test Accuracy"]

            for size in model_size:
                val_acc, test_acc = size_data.get(size, ("N/A", "N/A"))
                val_row.append(val_acc)
                test_row.append(test_acc)

            writer.writerow(val_row)
            writer.writerow(test_row)
else:
    rows = collect_results(configs, datasets, dims, model_size, False)

    # Original flat per-row write for no model_size grouping
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Dataset", "Dim", "Model Size", "Val Accuracy", "Test Accuracy", "Attack", "Epsilon", "Adversarial acc@1"])
        writer.writerows(rows)

print(f"✅ Results written to: {output_file}")
