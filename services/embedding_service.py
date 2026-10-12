import os
import torch
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)


# ===================== 设备自动选择 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载本地Qwen3-Embedding-0.6B模型
_model = None
def get_model():
    global _model
    if _model is None:
        MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/Qwen3-Embedding-0.6B")
        _model = SentenceTransformer(MODEL_PATH, device=device)
        logger.info(f"=== Embedding 模块启动 ===")
        logger.info(f"当前使用设备：{device}")
        logger.info(f"模型加载完成，路径：{MODEL_PATH}")
    return _model

def generate_embedding(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """
    对file_process输出的分块结果批量生成embedding
    :param chunks: 列表，每个元素是dict，包含 text, source, location 元数据
    :param batch_size: 批量大小，GPU推荐32~64；CPU建议8
    :return: 追加embedding向量后的chunk列表
    """
    if not chunks:
        return []

    texts = [item["text"] for item in chunks]
    start_time = time.time()
    model = get_model()

    # 批量编码
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # RAG检索推荐开启归一化
        show_progress_bar=True
    )

    end_time = time.time()
    total_token = sum(len(t) for t in texts)
    logger.info(f"\n批量向量化完成！")
    logger.info(f"总文本长度: {total_token} 字符")
    logger.info(f"耗时: {end_time - start_time:.2f} s")
    logger.info(f"速度: {total_token / (end_time - start_time):.2f} char/s")

    # 把向量放回原chunk字典，保留元数据
    result = []
    for chunk, vec in zip(chunks, embeddings):
        chunk["embedding"] = vec.tolist()
        result.append(chunk)
    return result

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

if __name__ == "__main__":
    # basicConfig 配置...
    model = get_model()
    test_chunks = [{"text": "这是一个测试句子", "meta": {}}]
    result = generate_embedding(test_chunks)
    print(f"向量维度: {len(result[0]['embedding'])}")

