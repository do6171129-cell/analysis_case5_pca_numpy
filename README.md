# 事例5：NumPyによる主成分分析（PCA）の実装と検証
## 概要

本事例では、NumPy のみを用いて主成分分析（PCA）を実装し、
主成分・主成分得点・寄与率が 線形代数の定義どおりに対応して得られること を検証する。

可視化・解釈・既存ライブラリとの比較は行わず、
定義 ⇄ 実装 ⇄ 出力対応の確認 にスコープを限定する。

## 目的

- NumPy のみで PCA を実装する
- 各定義量が数値計算として正しく対応していることを確認する
- PCA を「数値線形代数の手続き」として明示的に示す

## データ仕様
### 生成モデル

潜在因子を用いた線形生成モデルにより合成データを生成する。

$$
X = ZW + E
$$

- $Z \in \mathbb{R}^{n \times k}$：潜在因子（標準正規分布）
- $W \in \mathbb{R}^{k \times d}$：混合行列
- $E \in \mathbb{R}^{n \times d}$：加法ノイズ
- $X \in \mathbb{R}^{n \times d}$：観測データ


### パラメータ設定

本事例で用いるパラメータは以下のとおりである。

- サンプル数：500  
- 観測次元：5  
- 潜在因子数：2  
- ノイズ強度：0.3  
- 乱数シード：42

## 手法

本事例では、PCA を 定義に基づく数値計算手順 として実装する。
以下の操作が NumPy 上で数式どおり対応していることのみを確認する。

- 中心化

$$
X \in \mathbb{R}^{n \times d}
$$

$$
x_i \in \mathbb{R}^d \quad (i=1,\dots,n)
$$

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

$$
x_i^{(c)} = x_i - \mu
$$

$$
X_c =
\begin{pmatrix}
x_1^{(c)} \\
\vdots \\
x_n^{(c)}
\end{pmatrix}
$$

- 共分散行列

$$
\Sigma = \frac{1}{n} X_c^\top X_c
$$

- 固有値分解

$$
v_j \in \mathbb{R}^d,\quad \lambda_j \in \mathbb{R}
$$

$$
\Sigma v_j = \lambda_j v_j
$$

## スコープ外

本事例では、以下の内容は扱わない。

- 可視化（散布図、主成分空間での表示など）
- 主成分の解釈（寄与の意味づけ、特徴量重要度の議論など）
- scikit-learn 等の既存実装との比較・検証
- 前処理（標準化、欠損処理、外れ値処理など）
- 次元削減の適用判断やモデル選択（主成分数の選択など）

本事例の目的は、PCA を **定義に基づく数値計算手順として実装・検証すること**に限定される。

## ディレクトリ構成
```
analysis_case5_pca_numpy/
├── data/
│   ├── meta.json
│   └── X.csv
├── src/
│   ├── make_synthetic.py
│   └── pca.py
├── analysis_case5_pca_numpy.ipynb
├── .gitignore
└── .gitattributes
```


