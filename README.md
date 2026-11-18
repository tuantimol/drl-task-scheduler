# SLA-Aware DRL Task Scheduler for Edge–Cloud Environments
This repository provides the Python implementation of the framework proposed in the paper
“SLA-Aware Deep Reinforcement Learning for Adaptive Edge–Cloud Task Scheduling.”

# Overview
The code implements an SLA-aware, risk-sensitive deep reinforcement learning (DRL) scheduler for
dynamic task allocation in edge–cloud environments. The scheduling problem is formulated as a
sequential decision-making process, where the agent learns to minimise SLA violations while
optimising latency, task completion, and energy efficiency. A key contribution is the integration of an
explicit SLA Violation Risk Score (SVRS) into the DRL state and reward design, enabling proactive and
safe scheduling decisions.

## Hướng dẫn cài đặt và chạy dự án

**Bước 1: Kiểm tra Python và cài công cụ cần thiết**
Đảm bảo máy tính đang dùng Python từ bản **3.9** đến **3.12**, sau đó cài đặt thêm `venv`, `pip` và `git` nếu chưa có:
```bash
python3 --version
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

**Bước 2: Tải code về máy**
Clone dự án và di chuyển vào thư mục chứa mã nguồn (đảm bảo thấy các file `main.py`, `env.py`, `model.py`, `requirements.txt`,... Từ đây mọi lệnh đều chạy trong thư mục này):
```bash
cd ~
git clone [https://github.com/tuantimol/drl-task-scheduler.git](https://github.com/tuantimol/drl-task-scheduler.git)
cd drl-task-scheduler
ls
```

**Bước 3: Tạo môi trường ảo và cài thư viện**
*(Lưu ý: Sau khi kích hoạt, đầu dòng lệnh sẽ có chữ `(venv)` và bạn cần chạy lại `source venv/bin/activate` mỗi khi mở terminal mới. Quá trình cài `torch` có thể mất vài phút; nếu `gym` báo lỗi thì có thể bỏ qua vì `main.py` không dùng đến).*
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

**Bước 4: Kiểm tra cài đặt**
Chạy lệnh kiểm tra nhanh, nếu màn hình in ra `OK` kèm số phiên bản PyTorch là hoàn tất:
```bash
python -c "import torch, numpy, matplotlib; print('OK', torch.__version__)"
```

**Bước 5: Mở Jupyter Notebook**
```bash
jupyter notebook
```

# 🔹 Key Features

SLA Violation Risk Score (SVRS)–based state representation


Tier-aware SLA handling (Gold / Silver / Bronze tasks)


Hybrid Conv1D–GRU DRL architecture


SLA-aware reward shaping and action pruning


Closed-loop feedback using runtime SLA monitoring


Simulation-based and reproducible evaluation


Jupyter Notebook for experiments and visualization


# 📁 Code Structure
DRL_SLA_Task_Scheduler/

│
├── env.py        # Edge–cloud environment and workload simulation

├── model.py      # DRL model (Conv1D–GRU network architecture)

├── train.py      # Training logic for the DRL agent

├── test.py       # Testing and evaluation script

├── utils.py      # Utility functions (config, logging, metrics)

│
├── DRL_SLA_Task_Scheduler_Notebook.ipynb  

# Notebook for:

│                 # - Running experiments

│                 # - Showing outputs

│                 # - Graphs and results

│                 # - Step-by-step explanation

│                 # - Paper figures


└── README.md          # Project documentation

└── requirements.txt          # Required packages



# ⚙️ Usage

1. Train the model
python train.py

2. Test / evaluate the trained model
python test.py

3. Run experiments and visualize results

Open the notebook:

DRL_SLA_Task_Scheduler_Notebook.ipynb


This notebook is used for:

Running experiments

Plotting graphs

Showing outputs

Step-by-step explanation

Generating figures for the research paper

# Evaluation Scope
All results are obtained via high-fidelity simulation, as described in the paper. Real-world
deployment and testbed validation are considered future work.

# Reference
If you use this code, please cite the corresponding research paper.

