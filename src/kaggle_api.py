import json
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def init():
    os.makedirs("/root/.kaggle", exist_ok=True)
    shutil.copy("/kaggle/kaggle.json", "/root/.kaggle")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)


def date():
    day = datetime.today().day
    month = datetime.today().month
    date = f"{month:02}{day:02}"
    return date


def download_comp_datasets(comp_name):
    command = f"kaggle competitions download {comp_name} --path /kaggle/input/{comp_name}/"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
    shutil.unpack_archive(f"/kaggle/input/{comp_name}/{comp_name}.zip", f"/kaggle/input/{comp_name}/")
    os.remove(f"/kaggle/input/{comp_name}/{comp_name}.zip")


def download_datasets(dataset):
    savedir = dataset.split("/")[-1]
    command = f"kaggle datasets download {dataset} --unzip --quiet --path /kaggle/input/{savedir}"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)


def create_datasets(userid, folder):
    title = folder.rstrip("/").split("/")[-1].replace("_", "-")
    print(title)
    with open("/kaggle/src/dataset-metadata.json", "r") as js:
        dict_json = json.load(js)
    dict_json["title"] = title
    dict_json["id"] = f"{userid}/{title}"
    with open(f"{folder}/dataset-metadata.json", "w") as js:
        json.dump(dict_json, js, indent=4)

    command = f"kaggle datasets create -p {folder} --quiet --dir-mode zip"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)


def pull_kernel(kernel, path="./"):
    os.makedirs("/kaggle/reference/", exist_ok=True)
    fname = kernel.rstrip("/").split("/")[-1]
    command = f"kaggle kernels pull {kernel} --path /tmp"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
    shutil.move(f"/tmp/{fname}.ipynb", f"/kaggle/reference/{fname}.ipynb")


def push_kernel(userid, path, datasets=[], comp="", random_suffix=True):
    shutil.copy(path, "/tmp/tmp.ipynb")
    fname = Path(path)
    rand = random.randint(1000, 9999) if random_suffix else ""
    with open("/kaggle/src/kernel-metadata.json", "r") as js:
        dict_json = json.load(js)
    dict_json["id"] = f"{userid}/{fname.stem}{rand}"
    dict_json["title"] = f"{fname.stem}{rand}"
    dict_json["code_file"] = str(fname)
    dict_json["competition_sources"] = comp
    dict_json["enable_gpu"] = "true"
    dict_json["dataset_sources"] = datasets
    dict_json["language"] = "python"
    dict_json["kernel_type"] = "notebook"
    os.chdir("/")
    with open("/kernel-metadata.json", "w") as js:
        json.dump(dict_json, js, indent=4)
    command = "kaggle kernels push"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
    os.remove("/kernel-metadata.json")


def pull_model(url: str):
    owner = url.split("/")[4].lower()
    model_slug = url.split("/")[5].lower()
    framework = url.split("/")[7].lower()
    instance_slug = url.split("/")[9].lower()
    version_number = url.split("/")[11].lower()

    model = f"{owner}/{model_slug}/{framework}/{instance_slug}/{version_number}"
    print(model)
    save_path = f"/kaggle/input/{model}"
    print(save_path)
    command = f"kaggle models instances versions download {model} --untar -p {save_path}"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)


def submission(comp_name, file_path, message="submission"):
    command = f"kaggle competitions submit {comp_name} -f {file_path} -m {message}"
    try:
        subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
