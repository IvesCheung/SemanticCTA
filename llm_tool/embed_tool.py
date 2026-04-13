import requests
import json
from typing import List, Union
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================
# 本地模型配置
# 将 USE_LOCAL_MODEL 设为 True 即可切换为本地推理，避免 API 限速/延迟。
# 对于 modelname2path 中没有本地路径的模型（如 qwen3-embedding-4b），
# 无论该开关状态如何，仍会自动回退到 API 模式。
# ============================================================

#: 全局开关：True = 优先使用本地模型；False = 始终走 API
USE_LOCAL_MODEL: bool = True

#: 本地模型运行的设备，如 "cuda:0"、"cuda:1"、"cpu"
LOCAL_MODEL_DEVICE: str = "cuda:0"

#: 模型名称到本地权重目录的映射（仅 qwen3-embedding-0.6b 有本地路径）
modelname2path = {
    "qwen3-embedding-0.6b": "./model/qwen3-0.6B-embedding",
}

# 内部单例缓存，避免重复加载浪费显存：{model_name: (tokenizer, model)}
_local_model_cache: dict = {}

MAX_PROCESS_NUM = 256
TARGET_MODEL = 'qwen3-embedding-4b'
AuthorizationList = [
    'Bearer MAASe95b98cdd06a4b74bb3154ae31421cd3',
    'Bearer QSTab740dd3b95e5382bd46f837ca483944',
    'Bearer MAASe95b98cdd06a4b74bb3154ae31421cd3',
    'Bearer QST7ccaa7bc5e1c280724f48db6b119a54c',
    'Bearer MAASe95b98cdd06a4b74bb3154ae31421cd3'
]
EmbeddingModel = {
    'qwen3-embedding-0.6b': {
        'url': "http://redservingapi.devops.xiaohongshu.com/v1/embeddings",
        'model': "qwen3-embedding-0.6b",
        'encoding_format': "float"
    },
    'qwen3-embedding-4b': {
        'url': "https://aimi.devops.xiaohongshu.com/aimi-emb-qwen3/v1/embeddings",
        'model': "qwen3-embedding-4b",
        'encoding_format': "float"
    }
}


class EmbeddingAPIError(Exception):
    """Embedding 接口语义或结构异常"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1), reraise=True)
def _fetch_embedding_batch(model: str,
                           batch: List[str],
                           timeout: int = 10) -> List[List[float]]:
    """
    单批次请求（带自动重试）
    """
    url = EmbeddingModel[model]['url']
    headers = {'Content-Type': 'application/json'}
    if model == 'qwen3-embedding-0.6b':
        headers['Authorization'] = AuthorizationList[np.random.randint(
            0, len(AuthorizationList))]

    payload = {
        "model": model,
        "input": batch,
        "encoding_format": "float"
    }

    try:
        if model == 'qwen3-embedding-0.6b':
            resp = requests.post(url, data=json.dumps(
                payload), headers=headers, timeout=timeout)
        else:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EmbeddingAPIError(f"HTTP请求失败: {e}") from e

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise EmbeddingAPIError(f"响应JSON解析失败: {e}") from e

    if 'data' not in data or not data['data']:
        raise EmbeddingAPIError(f"响应缺少 data 字段或为空: {data}")
    if 'embedding' not in data['data'][0]:
        raise EmbeddingAPIError(f"响应缺少 embedding 字段: {data}")

    embeddings = [item['embedding'] for item in data['data']]

    cleaned = []
    for emb in embeddings:
        if any(np.isnan(x) for x in emb):
            emb = [0.0 if (isinstance(x, float) and np.isnan(x))
                   else x for x in emb]
        cleaned.append(emb)
    return cleaned


# ============================================================
# 本地模型推理（懒加载，单例）
# ============================================================

def _get_or_load_local_model(model_name: str):
    """懒加载本地 Embedding 模型，同一进程内只加载一次，节省显存。"""
    if model_name in _local_model_cache:
        return _local_model_cache[model_name]

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(
            "本地模型推理需要 torch 和 transformers，请先安装：pip install torch transformers"
        ) from e

    model_path = modelname2path[model_name]
    print(
        f"[embed_tool] 加载本地模型 {model_name} from {model_path} -> {LOCAL_MODEL_DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16)
    model.to(LOCAL_MODEL_DEVICE)
    model.eval()
    _local_model_cache[model_name] = (tokenizer, model)
    print(f"[embed_tool] 本地模型加载完成")
    return tokenizer, model


def _last_token_pool(last_hidden_states, attention_mask):
    """取每个序列最后一个有效 token 的隐状态（适配 Qwen3-Embedding 架构）。"""
    # 如果是左填充，直接取最后一列
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    import torch
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _fetch_embedding_batch_local(model_name: str, batch: List[str]) -> List[List[float]]:
    """使用本地模型对单批文本做 embedding 推理。"""
    import torch
    import torch.nn.functional as F

    tokenizer, model = _get_or_load_local_model(model_name)

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(LOCAL_MODEL_DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = _last_token_pool(
            outputs.last_hidden_state, encoded["attention_mask"]
        )
    embeddings = F.normalize(embeddings, p=2, dim=1)

    cleaned = []
    for emb in embeddings.cpu().float().numpy():
        emb_list = emb.tolist()
        if any(np.isnan(x) for x in emb_list):
            emb_list = [0.0 if np.isnan(x) else x for x in emb_list]
        cleaned.append(emb_list)
    return cleaned


# ============================================================
# 公共接口
# ============================================================

def get_embeddings(texts: List[str], model: str = TARGET_MODEL, batch_size=MAX_PROCESS_NUM) -> Union[List[List[float]], str]:
    """
    分批次获取文本 embedding。

    当 USE_LOCAL_MODEL=True 且 model 在 modelname2path 中时，使用本地 GPU 推理；
    否则（包括 qwen3-embedding-4b 等无本地路径的模型）退回 API 模式。

    Args:
        texts: 需要编码的文本列表
        model: 模型名称
        batch_size: 每批大小

    Returns:
        二维 embedding 列表
    """
    use_local = USE_LOCAL_MODEL and (model in modelname2path)

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        current_batch = i // batch_size + 1

        # 简单的文本进度条
        progress = current_batch / total_batches
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        # print(
        #     f'\r[{bar}] {progress:.0%} 批次 {current_batch}/{total_batches}', end='')
        try:
            if use_local:
                embeddings = _fetch_embedding_batch_local(model, batch)
            else:
                embeddings = _fetch_embedding_batch(model=model, batch=batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"\n批次 {current_batch} 最终失败: {e}")
            raise

    print()  # 换行
    return all_embeddings


def get_embeddings_by_model(texts, model, batch_size=64):
    all_embeddings = []
    model.eval()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"总文本数量: {len(texts)}", f"批次大小: {batch_size}")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        current_batch = i // batch_size + 1
        # 简单的文本进度条
        progress = current_batch / total_batches
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(
            f'\r[{bar}] {progress:.0%} 批次 {current_batch}/{total_batches}', end='')
        output = model(batch)
        output = output.tolist()
        all_embeddings.extend(output)
    return all_embeddings


def embedding_L2_normalization(embeddings: List[List[float]]):
    """
    L2归一化
    """
    embeddings_array = np.array(embeddings)
    # 计算每个embedding的L2范数
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    # 归一化
    embeddings_array = embeddings_array / norms
    return embeddings_array


def get_cosine_similarity_matrix(embeddings: List[List[float]]):
    """
    计算余弦相似度矩阵
    """
    embeddings_array = np.array(embeddings)
    # 计算余弦相似度
    cosine_sim = np.dot(embeddings_array, embeddings_array.T)
    return cosine_sim


def cosine_similarity(a, b):
    dot_product = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = sum(a[i] ** 2 for i in range(len(a))) ** 0.5
    norm_b = sum(b[i] ** 2 for i in range(len(b))) ** 0.5
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5
