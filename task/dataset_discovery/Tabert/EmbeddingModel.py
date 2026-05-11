"""
统一的 Embedding 模型接口
支持 Tabert 和 Qwen 两种 embedding 模型
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

# 导入 embed_tool
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

try:
    from llm_tool.embed_tool import get_embeddings
    from add_profilling import encode_column
except ImportError:
    print("Warning: Could not import embed_tool, Qwen embedding may not work")
    get_embeddings = None


class EmbeddingModelBase(ABC):
    """Embedding 模型的抽象基类"""

    @abstractmethod
    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """将文本列表编码为 embedding

        Args:
            texts: 文本列表

        Returns:
            二维 embedding 列表，shape: (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def encode_columns(self, table_id, headers: List[str], data: List[List[str]], **kwargs) -> np.ndarray:
        """将表的列编码为 embedding

        Args:
            headers: 列名列表
            data: 表数据 (List[List[str]])

        Returns:
            列的 embedding 矩阵，shape: (num_columns, embedding_dim)
        """
        pass


class TabertEmbedding(EmbeddingModelBase):
    """Tabert embedding 模型"""

    def __init__(self, model_path: str = "./model/tabert_base_k3/model.bin"):
        """初始化 Tabert 模型

        Args:
            model_path: Tabert 模型路径
        """
        from table_bert.table import Table, Column
        from table_bert.table_bert import TableBertModel
        from transformers import BertTokenizer

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tabert_model = TableBertModel.from_pretrained(model_path)
        self.tabert_model.to(self.device)
        self.tabert_model.eval()
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.Table = Table
        self.Column = Column

    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """Tabert 不直接支持纯文本编码，这里返回占位符"""
        raise NotImplementedError(
            "Tabert does not support direct text encoding")

    def encode_columns(self, table_id, headers: List[str], data: List[List[str]],
                       mask_header: bool = False, **kwargs) -> np.ndarray:
        """将表的列编码为 embedding

        Args:
            headers: 列名列表
            data: 表数据
            mask_header: 是否 mask header

        Returns:
            列的 embedding 矩阵
        """
        column_list = []
        value_1 = data[0] if len(data) >= 1 else [''] * len(headers)

        for i in range(len(headers)):
            column_list.append(
                self.Column(
                    headers[i].strip() if not mask_header else '##',
                    'text',
                    value_1[i]
                )
            )

        table = self.Table(
            id="temp_table",
            header=column_list,
            data=data if len(data) >= 1 else [['']*len(headers)]
        ).tokenize(self.tabert_model.tokenizer)

        context_tokens = self.tabert_model.tokenizer.tokenize("table")

        with torch.no_grad():
            context_encoding, column_encoding, info = self.tabert_model.encode(
                contexts=[context_tokens],
                tables=[table]
            )

        # column_encoding shape: (1, num_columns, embedding_dim)
        # (num_columns, embedding_dim)
        embeddings = column_encoding[0].cpu().numpy()
        return embeddings


class QwenEmbedding(EmbeddingModelBase):
    """Qwen embedding 模型"""

    def __init__(self, model_name: str = 'qwen3-embedding-4b', batch_size: int = 256, local_model_path: str = None, base_model_path: str = None):
        """初始化 Qwen embedding 模型

        Args:
            model_name: Qwen 模型名称（用于远程API）
            batch_size: 批次大小
            local_model_path: 本地训练好的模型权重文件路径（.pth文件）
            base_model_path: 基础模型路径（用于初始化ContrastiveEmbeddingModel结构）
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = 3584  # Qwen embedding 维度
        self.local_model_path = local_model_path
        self.base_model_path = base_model_path
        self.local_model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 如果提供了本地模型路径，加载本地训练的模型
        if local_model_path and os.path.exists(local_model_path):
            print(f"Loading local trained model from {local_model_path}")
            try:
                # 导入ContrastiveEmbeddingModel
                import importlib.util
                contrastive_path = os.path.join(
                    root_dir, 'taqwen', 'contrastive_learning_profilling.py'
                )
                spec = importlib.util.spec_from_file_location(
                    "contrastive_module", contrastive_path
                )
                contrastive_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(contrastive_module)
                ContrastiveEmbeddingModel = contrastive_module.ContrastiveEmbeddingModel

                # 确定基础模型路径
                if base_model_path is None:
                    # 尝试从常见位置推断
                    base_model_path = './model/qwen3-0.6B-embedding'
                    if not os.path.exists(base_model_path):
                        raise ValueError(
                            "base_model_path not provided and default path not found. "
                            "Please specify base_model_path for ContrastiveEmbeddingModel."
                        )

                print(
                    f"Initializing ContrastiveEmbeddingModel with base model: {base_model_path}")
                self.local_model = ContrastiveEmbeddingModel(
                    base_model=base_model_path,
                    device=self.device
                )

                # 加载训练好的权重
                self.local_model.load_state_dict(torch.load(
                    local_model_path, map_location=self.device))
                self.local_model.to(self.device)
                self.local_model.eval()

                # 更新embedding维度
                # self.embedding_dim = self.local_model.embedding_dim
                print(f"Local model loaded successfully on {self.device}")
                # print(f"Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading local model: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to remote API")
                self.local_model = None
        elif local_model_path:
            print(
                f"Warning: Local model path {local_model_path} does not exist")
            print("Falling back to remote API")

        # 如果没有本地模型，确保embed_tool可用
        if self.local_model is None and get_embeddings is None:
            raise ImportError(
                "embed_tool module not available and no local model provided")

    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """将文本列表编码为 embedding

        Args:
            texts: 文本列表

        Returns:
            二维 embedding 列表
        """
        # 如果有本地模型，使用本地模型编码
        if self.local_model is not None:
            all_embeddings = []
            self.local_model.eval()

            with torch.no_grad():
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:min(i + self.batch_size, len(texts))]

                    # 直接调用ContrastiveEmbeddingModel的forward方法
                    batch_embeddings = self.local_model(batch_texts)

                    # 转换为列表
                    batch_embeddings = batch_embeddings.cpu().numpy().tolist()
                    all_embeddings.extend(batch_embeddings)

            return all_embeddings
        else:
            # 使用远程API
            embeddings = get_embeddings(
                texts=texts,
                model=self.model_name,
                batch_size=self.batch_size
            )
            return embeddings

    def encode_columns(self, table_id, headers: List[str], data: List[List[str]],
                       sample_rows: int = 3, profilling_data={}, **kwargs) -> np.ndarray:
        """将表的列编码为 embedding

        构造每列的文本表示，包括列名和样本值
        使用更丰富的文本表示来提高语义理解

        Args:
            headers: 列名列表
            data: 表数据

        Returns:
            列的 embedding 矩阵，shape: (num_columns, embedding_dim)
        """
        texts = []
        mask_header = kwargs.get('mask_header', False)
        for col_idx, header in enumerate(headers):
            # 构造列的文本表示：列名 + 样本值
            # 提供更多的上下文信息来帮助模型理解列的语义
            texts.append(encode_column(
                table_name=table_id,
                header_name=header if not mask_header else mask_header,
                col_type=profilling_data.get(
                    header, {}).get('__type__', 'text'),
                # 这里必须取到对应列的样本值
                profile=profilling_data.get(header, {}),
                column_samples=[row[col_idx] for row in data],
            ))

        # 批量编码
        embeddings = self.encode_text(texts)

        return np.array(embeddings, dtype=np.float32)


class BGEEmbedding(EmbeddingModelBase):
    """BGE embedding 模型（via SentenceTransformers，纯本地推理）"""

    def __init__(self, model_path: str = "model/bge-large-en", batch_size: int = 256, **kwargs):
        """初始化 BGE embedding 模型

        Args:
            model_path: 本地模型路径，如 "model/bge-large-en"
            batch_size: 批次大小
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.model_path = model_path
        self.batch_size = batch_size

        print(f"Loading BGE model from: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"BGE model loaded. Embedding dimension: {self.embedding_dim}")

    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """将文本列表编码为 embedding（L2 归一化）

        Args:
            texts: 文本列表

        Returns:
            二维 embedding 列表
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_columns(self, table_id, headers: List[str], data: List[List[str]],
                       sample_rows: int = 3, profilling_data={}, **kwargs) -> np.ndarray:
        """将表的列编码为 embedding

        Args:
            table_id: 表名
            headers: 列名列表
            data: 表数据 (List[List[str]])
            sample_rows: 采样行数（已在 data 中体现）
            profilling_data: profilling 数据字典
            **kwargs: mask_header 等可选参数

        Returns:
            列的 embedding 矩阵，shape: (num_columns, embedding_dim)
        """
        texts = []
        mask_header = kwargs.get('mask_header', False)
        for col_idx, header in enumerate(headers):
            texts.append(encode_column(
                table_name=table_id,
                header_name=header if not mask_header else mask_header,
                col_type=profilling_data.get(
                    header, {}).get('__type__', 'text'),
                profile=profilling_data.get(header, {}),
                column_samples=[row[col_idx] for row in data],
            ))

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)


class EmbeddingModelFactory:
    """Embedding 模型工厂类"""

    _models = {
        'tabert': TabertEmbedding,
        'qwen': QwenEmbedding,
        'bge': BGEEmbedding,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs) -> EmbeddingModelBase:
        """创建 embedding 模型

        Args:
            model_type: 模型类型 ('tabert', 'qwen', 'bge')
            **kwargs: 传递给模型的参数

        Returns:
            Embedding 模型实例

        Note:
            当 model_type='qwen' 且 model_name 中含有 'bge' 时，
            自动路由到 BGEEmbedding，无需改动上层调用代码。
            例如：qwen_model="model/bge-large-en" 即可直接使用 BGE。
        """
        # 自动检测 BGE：model_name 包含 'bge' 时路由到 BGEEmbedding
        if model_type == 'qwen' and 'bge' in kwargs.get('model_name', '').lower():
            print(
                f"[EmbeddingModelFactory] Detected BGE model, routing to BGEEmbedding.")
            return BGEEmbedding(
                model_path=kwargs.get('model_name'),
                batch_size=kwargs.get('batch_size', 256),
            )

        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = cls._models[model_type]
        return model_class(**kwargs)

    @classmethod
    def register(cls, model_type: str, model_class: type) -> None:
        """注册自定义 embedding 模型

        Args:
            model_type: 模型类型
            model_class: 模型类
        """
        cls._models[model_type] = model_class


if __name__ == '__main__':
    # 测试 Qwen embedding
    print("Testing Qwen Embedding...")
    try:
        qwen_model = EmbeddingModelFactory.create(
            'qwen',
            model_name='qwen3-embedding-4b',
            batch_size=32
        )

        # 测试文本编码
        test_texts = ["Column: Name", "Column: Age", "Column: Email"]
        embeddings = qwen_model.encode_text(test_texts)
        print(
            f"Text embeddings shape: {len(embeddings)} x {len(embeddings[0])}")

        # 测试列编码
        headers = ["Name", "Age", "Email"]
        data = [
            ["Alice", "25", "alice@example.com"],
            ["Bob", "30", "bob@example.com"],
        ]
        col_embeddings = qwen_model.encode_columns(headers, data)
        print(f"Column embeddings shape: {col_embeddings.shape}")

    except Exception as e:
        print(f"Error: {e}")
