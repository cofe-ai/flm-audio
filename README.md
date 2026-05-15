# FLM-Audio（量化AWQ/GPTQ版本）推理运行说明

本指南提供运行FLM-Audio量化版本（AWQ/GPTQ）的详细步骤。

---

## 环境配置（量化模型支持）
以下步骤配置支持AWQ和GPTQ量化的服务器环境：

### 1. 安装`uv`包管理器
`uv`是快速Python包管理器，用于环境和依赖管理：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
验证安装：
```bash
uv --version
```

### 2. 克隆`flm-audio`仓库
```bash
git clone -b quantized https://github.com/cofe-ai/flm-audio.git
cd flm-audio
git submodule update --init --recursive
```

### 3. 创建Python 3.12虚拟环境
```bash
uv venv -p 3.12
```
将创建包含Python 3.12的`.venv`目录，可选择性激活（uv会自动检测虚拟环境）：
```bash
source .venv/bin/activate  # Linux/macOS
```

### 4. 安装PyTorch 2.5.1（CUDA 12.1）
安装支持CUDA 12.1的PyTorch版本，用于GPU推理：
```bash
uv pip install torch==2.5.1 --torch-backend=cu121
```

### 5. 安装服务器依赖
从requirements文件安装核心服务器依赖：
```bash
uv pip install -r requirements-server.txt
```

### 6. 安装AutoAWQ（AWQ量化必选）
安装本地预下载的AutoAWQ包，支持AWQ量化：
```bash
uv pip install deps/AutoAWQ --no-build-isolation
```

### 7. 安装AutoAWQ Kernels（AWQ GEMV量化必选）
安装AutoAWQ推理所需的内核：
```bash
uv pip install deps/AutoAWQ_kernels --no-build-isolation
```

### 8. 安装AutoGPTQ（GPTQ量化必选）
安装本地预下载的AutoGPTQ包，支持GPTQ量化：
```bash
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
uv pip install deps/AutoGPTQ --no-build-isolation
```

---

## 启动服务器
所有依赖安装完成后，启动FLM-Audio服务器：
```bash
python -m flmaudio.server --port 8990 --model-path /path/to/quantized/model
```
服务器将加载量化模型，默认监听8990端口。

---

## 运行客户端
提供两种客户端：基于Gradio的Web界面和CLI客户端，均连接到运行中的服务器（默认`http://localhost:8990`）。

### Web界面（Gradio）
1. 安装客户端依赖：
   ```bash
   uv pip install -r requirements-clientgui.txt
   ```
2. 启动Web界面：
   ```bash
   python -m flmaudio.client_gradio --url http://localhost:8990
   ```
   在浏览器中访问（默认为`http://localhost:50000`）。

### CLI客户端
1. 安装CLI依赖：
   ```bash
   uv pip install -r requirements-clientcli.txt
   ```
2. 启动CLI客户端：
   ```bash
   python -m flmaudio.client --url http://localhost:8990
   ```

## 许可证
FLM-Audio采用[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)许可，但`third_party/moshi`下的Python代码采用[MIT License](https://opensource.org/license/mit/)。本项目仅用于研究，需遵守适用法律。商业使用请联系我们。
