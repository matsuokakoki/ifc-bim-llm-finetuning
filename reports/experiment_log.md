# 実験記録：重複除去後の Gemma 3 1B LoRA

## 目的と範囲

IFC/BIM の質問応答に Gemma 3 1B IT を適応させ、学習と評価の完全一致質問を除いた後も、別形式の評価データで学習前より指標が改善するか調べた。元の Kaggle 実験の保存出力をまとめた記録であり、このリポジトリ上での再学習結果ではない。

## データと重複除去

| 段階 | 件数 |
| --- | ---: |
| Alpaca 形式の学習元 | 42,680 |
| 評価質問と正規化完全一致する学習側の削除行 | 657 |
| 重複除去後 | 42,023 |
| 学習 | 33,618 |
| Validation | 8,405 |
| 別形式の評価セット | 900 |

学習側の一致657行は質問と回答の両方が一致していた。Unicode NFKC、casefold、空白の正規化をした質問で比較し、学習元から削除してから質問単位で学習・Validationに分割した。学習と評価、Validation と評価、学習と Validation の正規化完全一致質問は、分割後いずれも0件。評価900件は変更していない。657は学習側の行数であり評価900件の73%という意味ではない。評価内には同一質問が2行あり、同一の質問・回答ペアは0行だった。

近似重複や言い換えは除外対象外。同じ公開者の同じIFC/BIMドメインのデータを使用したため、独立した外部ベンチマークでもない。重複の集計原本は [data_contamination_report.csv](../results/data_contamination_report.csv)。

## 学習設定

| 項目 | 値 |
| --- | --- |
| ベース | Gemma 3 1B IT |
| 方法 | 通常の LoRA、回答部分のみ loss 対象 |
| rank / alpha / dropout | 16 / 32 / 0.05 |
| 対象層 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| batch / gradient accumulation | 4 / 1 |
| epoch / learning rate | 2 / 1e-4 |
| 精度 / 最大長 | bfloat16 / 1536 tokens |
| 乱数 seed | 42 |

42,023件のトークン長について、1024 tokens では全文が入る割合が32.14%、1536では100%。元の学習は16,810 step、約8,104秒。GPUメモリの最大 reserved は約18.44 GB。これは当該実行での記録で、他環境の必要量を保証しない。

## 数値結果

| 指標 | 学習前 | 学習後 | 対象・解釈 |
| --- | ---: | ---: | --- |
| 回答部分の loss | 3.9157 | 0.0662 | Validation 8,405件。同じ学習元データセット |
| 回答部分の loss | 5.0349 | 1.4989 | 別形式の評価900件 |
| ROUGE-1 | 0.1370 | 0.2040 | 生成した評価50件 |
| ROUGE-L | 0.0973 | 0.1636 | 同50件 |
| BLEU | 0.0174 | 0.0502 | 同50件 |

ROUGE/BLEU は実験コード内の簡易実装で計算した同じ50件の相対比較。標準の評価実装や他研究の数値と直接比較しない。回答の表層一致を測るため、IFC式の正確性を保証しない。50件の質問・参照回答・生成回答は [比較CSV](../results/before_after_generation_gemma_subset.csv)、集計の抜粋は [metrics.csv](../results/metrics.csv) にある。

## 生成例から分かること

学習前の回答は一般的な長文に流れやすく、学習後は IFC 固有の語彙や式を直接出す傾向が強まった。ただし、比較CSVの id=0 では `CorrectEventTriggerType` を問われ、学習後の回答が `SIZEOF(IsTypedBy)` を使う別のルールを答えている。ほかに属性階層の取り違えや反復もある。一般質問「Are there any costs or licensing fees associated with using IFC?」に対しては、学習後の回答が費用を捏造しており、専門的な語彙への適応と内容の正しさを区別する必要がある。

50件への LLM-as-a-Judge の採点は未実施。採点用プロンプトの準備だけでは採点完了と扱わない。

## 成果物の範囲と出典

このリポジトリには集計CSVと50件の生成比較を収録する。生成比較に含まれる評価データの質問・参照回答は [Dietmar2020/ifc-bim-gemma3-subset-1k](https://huggingface.co/datasets/Dietmar2020/ifc-bim-gemma3-subset-1k)（MIT）由来。学習データは [Dietmar2020/ifc-bim-high-quality-alpaca](https://huggingface.co/datasets/Dietmar2020/ifc-bim-high-quality-alpaca)（Apache 2.0）。ベースモデルの利用条件は [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) を参照。モデル重みや学習データ全文は配布せず、元の提供元から取得する。

結果CSVは保存出力から確認した。学習スクリプトの冒頭説明には評価1,000件の記述があるが、今回の保存結果は900件。評価件数は保存結果の `gemma_eval_rows=900` を採用した。
