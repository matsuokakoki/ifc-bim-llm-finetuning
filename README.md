# IFC/BIM Gemma 3 1B LoRA 再実験

IFC/BIM の質問応答データで Gemma 3 1B Instruct を通常の LoRA で追加学習するための実験コードです。学習時は system/user 部分を loss 対象外にし、assistant の回答部分を学習します。また、Gemma 評価セットの質問と重なる Alpaca 学習データを除外する処理を含みます。

## コード

- [`notebooks/retrain_gemma3_lora.py`](notebooks/retrain_gemma3_lora.py) — Kaggle GPU 環境向けの再実験スクリプト
- LoRA rank 16、batch size 4、2 epochs の1条件を実行する設定です。HPO は行いません。

## 実行環境と入力

Kaggle Notebook で GPU accelerator を有効にし、`/kaggle/input` 以下に次のモデルと Hugging Face Datasets の保存形式のデータセットを追加してください。スクリプトはディレクトリ名の一部から入力を探索します。

- Gemma 3 1B Instruct（`gemma-3-1b-it` を含むモデルディレクトリ）
- IFC/BIM Alpaca 学習セット（`ifc-bim-high-quality-alpaca` を含むディレクトリ）
- IFC/BIM Gemma 評価セット（`ifc-bim-gemma3-subset-1k` を含むディレクトリ）

モデルとデータセットはこのリポジトリに含めていません。配布元の利用条件、Gemma の利用規約、Kaggle への追加方法を確認してください。実験コードは Kaggle のローカル GPU 環境で動かす想定で、実行・再現確認はこの追加時点では行っていません。

## 出力とデータ取り扱い

スクリプトは学習済み LoRA adapter、評価サマリー、生成例、曲線、データ重複検査レポートなどを `/kaggle/working` 以下に保存し、最後に ZIP 化します。出力には評価セットの質問・参照回答や生成結果が含まれる場合があります。データセットのライセンスと内容を確認するまで、出力物を公開・再配布しないでください。

入力データは作業用ディレクトリへコピーされます。既存の `/kaggle/working` 内の同名作業ディレクトリ／パッケージディレクトリはスクリプトが削除して作り直すため、Kaggle の使い捨て作業環境で実行してください。Weights & Biases はオフラインモードです。

## 結果について

このリポジトリには、再実験結果や実行済みの成功報告を含めていません。数値を報告する場合は、実際の実行ログ・生成物と評価条件を照合してください。
