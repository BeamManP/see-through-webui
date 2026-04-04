# See-through WebUI — ワンクリックインストールガイド

## 概要

アニメイラスト1枚から最大23レイヤーのセマンティック分解を行う
[See-through](https://github.com/shitagaki-lab/see-through) のローカルWebUI。

**対象**: Windows 10/11 + NVIDIA GPU (VRAM 8GB以上推奨)

---

## クイックスタート

### 1. インストール
`install.bat` をダブルクリックするだけ。以下が自動的に行われます：

1. Python 3.12 の確認（なければ自動ダウンロード＆インストール）
2. 仮想環境 (venv) の作成
3. PyTorch 2.8 + CUDA 12.8 のインストール
4. 依存パッケージのインストール
5. NF4量子化モデル (~3GB) のダウンロード

**所要時間**: 初回 15〜30分（回線速度による）

### 2. 起動
`run.bat` をダブルクリック → ブラウザが自動で開きます。

---

## ファイル構成

```
see-through/
├── install.bat          ← インストーラー（初回のみ）
├── run.bat              ← 起動ランチャー（毎回）
├── webui/
│   ├── README.md        ← このファイル
│   └── requirements.txt ← WebUI用依存パッケージ
├── tools/
│   └── webui.py         ← WebUI本体
├── venv/                ← 仮想環境（install.batで作成）
└── .hf_cache/           ← モデルキャッシュ
```

---

## 動作要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| OS | Windows 10 (64-bit) | Windows 11 |
| GPU | NVIDIA (VRAM 6GB) | NVIDIA (VRAM 10GB+) |
| メモリ | 8GB | 16GB |
| ストレージ | 15GB空き | 20GB空き |
| Python | 自動インストール | - |

### VRAM目安 (NF4モード)

| 解像度 | VRAM消費 |
|--------|----------|
| 512    | ~5GB     |
| 768    | ~5.5GB   |
| 1024   | ~7GB     |
| 1280   | ~9GB     |

VRAM 8GB環境では解像度768以下を推奨。

---

## トラブルシューティング

### 「Python が見つかりません」
→ install.bat が自動で Python 3.12 をインストールします。
  手動インストールの場合: https://www.python.org/downloads/

### 「CUDA error」
→ NVIDIA ドライバを最新版に更新してください。
  https://www.nvidia.com/drivers

### 「モデルのダウンロードが途中で止まる」
→ install.bat をもう一度実行すれば途中から再開します。

### 「VRAM不足 / Out of Memory」
→ WebUIの解像度スライダーを下げてください（512〜768推奨）。
