#"mean"         平均ベクトル μ（中心化に使用）
#"eigvals"      固有値（降順）
#"components"   主成分（固有ベクトルを列に持つ行列 V）
#"scores"       主成分得点 T = Xc @ V
#"explained_ratio"   寄与率


import numpy as np

def pca(X: np.ndarray) -> dict:
    # 中心化
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    #分散・共分散行列の作成
    n=X.shape[0]
    sigma = Xc.T@Xc/n

    #固有値、固有ベクトルを取る
    eigvals, eigvecs = np.linalg.eigh(sigma)

    #並び替え
    idx = np.argsort(eigvals)[::-1]  # 大きい順の順番メモ
    eigvals = eigvals[idx]  # 固有値を並び替え
    components = eigvecs[:, idx]  # 対応する列を同じ順で並び替え

    #主成分得点
    scores = Xc @ components

    #寄与率
    explained_ratio = eigvals / eigvals.sum()

    # （次：共分散へ続く）
    return {
        "mean": mean,
        # 仮置き（次で埋める）
        "eigvals": eigvals,
        "components": components,
        "scores": scores,
        "explained_ratio": explained_ratio,
    }
