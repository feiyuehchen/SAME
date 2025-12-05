# TitaNet + Dual-Bank Memory Network Technical Report

## 1. 專案概述 (Project Overview)

### 1.1 研究目標
本專案旨在開發一個強健的 **Memory-Augmented Supervised Learning System** 用於音訊 Deepfake 檢測。系統結合了：
- **TitaNet**: NVIDIA 的聲紋辨識模型（作為特徵提取器）。
- **Dual-Bank Memory Network**: 雙記憶庫架構，分別對應「真實語音 (Bonafide)」與「偽造語音 (Spoof)」。
- **Optimal Transport (OT)**: 利用 Sinkhorn 演算法確保特徵均勻映射到記憶庫，解決 Mode Collapse 問題。
- **Discriminative Learning**: 引入 OC-Softmax 與 Sparse Attention 強化特徵的區別性。

### 1.2 理論基礎
**Dual-Bank Discriminative Paradigm**:
- **架構創新**: 不同於傳統單一記憶庫，我們建立了 `Bonafide Bank` 與 `Spoof Bank`。
- **訓練機制**:
    - **Attraction**: 真實語音應靠近 Bonafide Bank，偽造語音應靠近 Spoof Bank。
    - **Repulsion**: 真實語音應遠離 Spoof Bank，偽造語音應遠離 Bonafide Bank。
- **推理機制**:
    - 使用 **雙邊重構誤差差值** (`Error_Spoof - Error_Real`) 作為最終分數。
    - 分數越高表示越像真實語音 (因為 Real Error 低且 Spoof Error 高)。

---

## 2. 核心演算法 (Core Algorithms)

### 2.1 Dual-Bank Memory Network (雙記憶庫網路)
**概念**: 建立兩組獨立的可學習原型 (Prototypes)，分別捕捉真實與偽造語音的特徵分佈。
- **結構**: 
    - $M_{real} \in \mathbb{R}^{K \times D}$ (例如 $64 \times 192$)
    - $M_{spoof} \in \mathbb{R}^{K \times D}$
- **Sparse Attention (稀疏注意力機制)**:
    - 為了減少噪聲並專注於最相關的原型，我們只選取相似度最高的 Top-K 個 Slot 進行重構。
    - **步驟**:
        1. 計算 Query $z$ 與 Memory $M$ 的 Cosine Similarity。
        2. 選取 Top-K (例如 K=10) 的值與索引。
        3. 僅對這 Top-K 進行 Softmax 計算權重。
        4. 重構向量 $z_{recon} = \sum_{k \in TopK} w_k \cdot m_k$。

### 2.2 Optimal Transport (OT) with Sinkhorn Algorithm
**問題**: 傳統 Attention 容易導致 Mode Collapse，即大部分 Query 只映射到少數幾個 Memory Slot，導致其他 Slot 閒置 (Dead Slots)。
**解法**: 利用 OT 強制 Batch 內的樣本均勻分配到所有 Memory Slot。
**演算法 (Sinkhorn-Knopp)**:
1. **Logits 計算**: $L = \text{scale} \cdot \text{CosineSim}(z, M)$
2. **初始化**: $Q = \exp(L / \epsilon)$
3. **迭代歸一化**:
    - 行歸一化: $Q = Q / \sum_{row} Q$
    - 列歸一化: $Q = Q / \sum_{col} Q$
    - 重複迭代 (例如 3 次)。
4. **OT Loss**: 計算預測分佈與 OT 目標分佈 $Q$ 之間的 Cross-Entropy。
    - $L_{OT} = -\sum Q_{detached} \cdot \log(\text{Softmax}(L))$

### 2.3 One-Class Softmax (OC-Softmax)
**目的**: 在特徵空間中強制將真實語音壓縮得更緊密。
**機制**: 
- 學習一個中心 $c$。
- 對於真實語音，最大化其與 $c$ 的相似度（使其大於 $r_{real}$）。
- 對於偽造語音，最小化其與 $c$ 的相似度（使其小於 $r_{fake}$）。
- **公式**: 
    $$L_{oc} = \text{mean}(\text{Softplus}(\alpha \cdot (r_{real} - \text{score}_{bonafide}))) + \text{mean}(\text{Softplus}(\alpha \cdot (\text{score}_{spoof} - r_{fake})))$$

### 2.4 Entropy-based Diversity Loss
**目的**: 即使有 OT，仍需顯式地最大化 Attention 分佈的熵，以確保 Memory Slot 的利用率。
**公式**:
- 計算 Batch 內的平均 Attention: $\bar{w} = \frac{1}{B} \sum_{i=1}^B w_i$
- 最大化 Entropy (最小化負 Entropy): $L_{div} = \sum \bar{w} \log \bar{w}$

---

## 3. 模型架構與訓練 (Model Architecture & Training)

### 3.1 數據增強 (Data Augmentation)
為了防止模型過擬合特定的錄音環境或編解碼痕跡，我們採用雙重增強策略：

1.  **RawBoost (訊號級增強)**:
    - **LnL Convolutive Noise**: 模擬線性與非線性通道失真。
    - **ISD Additive Noise**: 模擬脈衝噪聲。
    - **SSI Additive Noise**: 模擬平穩加性噪聲。
2.  **Codec Augmentation (編解碼增強)**:
    - **MP3 / AAC**: 隨機對輸入音訊進行有損壓縮與解壓縮，強迫模型學習壓縮痕跡以外的特徵。

### 3.2 損失函數組合與詳細計算 (Loss Function Details)
總 Loss 由四個部分組成，權重分別由 $\lambda$ 參數控制：

$$L_{total} = \lambda_{recon} L_{recon} + \lambda_{OT} L_{OT} + \lambda_{OC} L_{OC} + \lambda_{div} L_{div}$$

#### 3.2.1 Dual Reconstruction Loss ($L_{recon}$)
此 Loss 負責驅動雙記憶庫的 Attraction（吸引）與 Repulsion（排斥）機制。我們希望樣本能被其所屬類別的記憶庫完美重構（誤差小），並無法被對立類別的記憶庫重構（誤差大）。

- **計算邏輯**:
    1. **Bonafide 樣本 ($y=0$)**:
        - **Attraction**: 最小化與 $M_{real}$ 的重構誤差 $E_{real}$。
        - **Repulsion**: 最大化與 $M_{spoof}$ 的重構誤差 $E_{spoof}$。這部分使用 Hinge Loss，只要誤差超過 `margin` (1.0) 即停止懲罰。
        - $L_{bonafide} = \text{mean}(E_{real}) + \text{mean}(\text{ReLU}(margin - E_{spoof}))$
    2. **Spoof 樣本 ($y=1$)**:
        - **Attraction**: 最小化與 $M_{spoof}$ 的重構誤差 $E_{spoof}$。
        - **Repulsion**: 最大化與 $M_{real}$ 的重構誤差 $E_{real}$。同樣使用 Hinge Loss。
        - $L_{spoof} = \text{mean}(E_{spoof}) + \text{mean}(\text{ReLU}(margin - E_{real}))$
    
    $$L_{recon} = L_{bonafide} + L_{spoof}$$

#### 3.2.2 Optimal Transport Loss ($L_{OT}$)
此 Loss 僅應用於樣本與其**對應**的記憶庫之間（Real 樣本 $\leftrightarrow$ Real Bank，Spoof 樣本 $\leftrightarrow$ Spoof Bank），目的是確保記憶庫的使用率均勻。

- **輸入**: 預測的分配機率 $P = \text{Softmax}(\text{logits})$ 與 Sinkhorn 算法計算出的最佳傳輸計劃 $Q$。
- **計算**:
    $$L_{OT} = -\sum_{i} Q_{detached}^{(i)} \cdot \log(P^{(i)})$$
- **意義**: 強迫模型輸出的相似度分佈（$P$）去逼近數學上最優的均勻分配（$Q$）。注意 $Q$ 必須 detach，我們不希望反向傳播影響 Sinkhorn 的迭代過程。

#### 3.2.3 One-Class Softmax Loss ($L_{OC}$)
此 Loss 作用於特徵空間 (Embedding Space)，而非記憶庫空間。目的是增強特徵的類別區分度。

- **定義**: 設定一個可學習的中心點 $C$ (Bonafide Center)。
- **計算**:
    - 計算特徵 $x$ 與中心 $C$ 的 Cosine Similarity: $s = \cos(x, C)$。
    - **Bonafide**: 希望 $s > r_{real}$ (例如 0.9)。
        - Loss term: $\text{Softplus}(\alpha \cdot (r_{real} - s))$
    - **Spoof**: 希望 $s < r_{fake}$ (例如 0.5)。
        - Loss term: $\text{Softplus}(\alpha \cdot (s - r_{fake}))$
- **意義**: 將真實語音緊密壓縮在以 $C$ 為中心的圓錐內，並將偽造語音推至圓錐外。

#### 3.2.4 Diversity Loss ($L_{div}$)
此 Loss 是一個輔助正則項，直接作用於 Attention Weights。

- **計算**:
    1. 計算 Batch 內所有樣本對記憶庫的平均關注度: $\bar{w} = \text{mean}_{batch}(w)$。
    2. 計算 $\bar{w}$ 的熵 (Entropy): $H(\bar{w}) = -\sum \bar{w} \log \bar{w}$。
    3. 我們希望熵最大化（分佈最均勻），因此最小化負熵。
    $$L_{div} = -H(\bar{w})$$
- **意義**: 防止模型只學會用某幾個特定的 "萬能 Slot" 來重構所有聲音，強迫所有 Slot 都被平均使用。

### 3.3 訓練流程
1.  **Input**: Raw Waveform (經過 Augmentation)。
2.  **Encoder**: TitaNet 提取 192-dim Embedding $z$。
3.  **Memory Interaction**:
    - $z$ 分別與 $M_{real}$ 和 $M_{spoof}$ 進行 Sparse Attention 交互。
    - 得到 $z_{recon\_real}$ 與 $z_{recon\_spoof}$。
4.  **OT Alignment**:
    - 僅對應類別的樣本進行 OT 計算 (例如真實樣本只對齊 $M_{real}$)。
5.  **Loss Calculation**: 計算上述四種 Loss 並反向傳播更新所有參數 (包含 Encoder 與 Memory)。

---

## 4. 評估與測試 (Evaluation)

### 4.1 評分標準 (Scoring)
不同於傳統只看重構誤差，Dual-Bank 架構使用差值分數：
$$\text{Score} = \text{Error}(z, M_{spoof}) - \text{Error}(z, M_{real})$$
- **邏輯**:
    - 真實語音應與 $M_{real}$ 很像 (誤差小)，與 $M_{spoof}$ 很不像 (誤差大) $\rightarrow$ 分數為正且大。
    - 偽造語音應與 $M_{spoof}$ 很像 (誤差小)，與 $M_{real}$ 很不像 (誤差大) $\rightarrow$ 分數為負且小。

### 4.2 數據集
- **ASVspoof 2019 LA**: 主要開發與測試集。
- **ASVspoof 2021 LA**: 用於測試模型的跨資料集泛化能力 (Robustness)。

### 4.3 指標
- **EER (Equal Error Rate)**: 錯誤接受率與錯誤拒絕率相等時的比率。
- **min t-DCF**: 結合 ASV 系統的綜合檢測成本函數。
