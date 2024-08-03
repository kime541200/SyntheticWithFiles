# 使用Unsloth對Llama 3.1進行超高效微調

> [原文連結](https://huggingface.co/blog/mlabonne/sft-llama3)

*mlabonne Maxime Labonne初學者指南：最新的監督微調技術*

Llama 3.1 的最新發布提供了具有驚人性能的模型，縮小了封閉源代碼和開源模型之間的差距。與其使用像 GPT-4o 和 Claude 3.5 這樣的通用、預先訓練好的語言模型，你可以根據自己的具體需求對 Llama 3.1 進行微調，以實現更好的性能和可定制性，並降低成本。

![00](./imgs/00.png)

在本文中，我們將提供一份全面性的監督微調概覽。接下來，我們將比較監督微調與提示工程，以了解何時使用監督微調是合理的，詳細介紹主要技術及其優缺點，並介紹重要概念，如LoRA超參數、存儲格式和聊天模板。最後，我們將在實踐中實現監督微調，通過在Google Colab中使用Unsloth對Llama 3.1 8B進行微調，達到最先進的優化效果。本文中使用的所有代碼都可以在[Google Colab](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN)和[LLM Course](https://github.com/mlabonne/llm-course)中找到。特別感謝Daniel Han回答我的問題。

## 🔧 監督微調（Supervised Fine-Tuning）

![01](./imgs/01.png)

監督微調（Supervised Fine-Tuning，SFT）是一種**改進和自定義**預先訓練語言模型（LLM）的方法。它涉及使用較小的指令和答案數據集來重新訓練基礎模型。主要目的是將一個基本的預測文本的模型轉變為一個可以遵循指令和回答問題的助手。SFT還可以增強模型的整體性能，添加新知識，或將其適應特定的任務和領域。微調後的模型可以選擇性地進行偏好對齊階段（見 [artical about DPO](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)以移除不想要的回應、修改其風格等。

下圖顯示了一個指令樣本。它包括一個系統提示來引導模型，一個用戶提示來提供任務，以及模型預期生成的輸出。你可以在[💾 LLM Datasets](https://github.com/mlabonne/llm-datasets) GitHub倉庫中找到一份高質量的開源指令數據集列表。

![02](./imgs/02.png)

在考慮監督微調（SFT）之前，我建議嘗試使用提示工程技術，如少**量示例提示（few-shot prompting）**或**增強生成檢索（retrieval augmented generation，RAG）**。在實踐中，這些方法可以解決許多問題，而無需進行微調，無論是使用封閉源代碼模型還是開放權重模型（例如Llama 3.1 Instruct）。如果這種方法無法達到您的目標（在質量、成本、延遲等方面），則當指令數據可用時，SFT就成為了一種可行的選擇。請注意，SFT還提供了其他好處，如額外的控制和自定義功能，以創建個人化的LLM。

然而，監督微調（SFT）也有其局限性。它在利用基礎模型已有的知識時效果最佳。學習完全新的信息，例如未知語言，可能會很具挑戰性，並導致更頻繁的幻覺。對於基礎模型未知的新領域，建議先在原始數據集上進行持續的預訓練。在另一端，指令模型（即已經微調過的模型）可能已經非常接近您的需求。例如，一個模型可能表現得非常好，但聲稱它是由OpenAI或Meta訓練的，而不是您。在這種情況下，您可能希望通過偏好對齊（preference alignment）稍微調整指令模型的行為。通過提供選擇和拒絕的樣本（約100到1000個樣本）來微調模型的行為，您可以強迫LLM說明您訓練了模型，而不是OpenAI。

。⚖️ 監督微調技術（SFT Techniques）

目前最受歡迎的三種監督微調技術是完整微調（full fine-tuning）、LoRA和QLoRA。

![03](./imgs/03.png)

**完整微調（Full fine-tuning）**是最直接的監督微調技術。它涉及使用指令數據集重新訓練預先訓練模型的所有參數。這種方法通常能夠提供最佳結果，但需要大量的計算資源（需要多個高端GPU來微調一個8B模型）。由於它修改了整個模型，因此也是最具破壞性的方法，可能會導致之前學習的技能和知識的災難性遺忘。

**低秩適應（Low-Rank Adaptation，LoRA）**是一種流行的參數高效微調技術。與其重新訓練整個模型，LoRA會凍結權重並在每個目標層引入小型適配器（低秩矩陣）。這使得LoRA能夠訓練的參數數量遠遠低於完整的微調（少於1％），從而減少了內存使用和訓練時間。這種方法是非破壞性的，因為原始參數被凍結，適配器可以在後續被切換或組合。

**QLoRA（量化感知低秩適應）**是LoRA的一個擴展，提供了更大的記憶體節省。它提供了高達33%的額外記憶體節省，相比於標準的LoRA，特別是在GPU記憶體受限的情況下。這種增加的效率是以訓練時間更長為代價的，QLoRA通常需要比標準LoRA多39%的時間來訓練。雖然QLoRA需要更多的訓練時間，但它在記憶體方面的顯著節省可以使其成為在GPU記憶體有限的場景中唯一可行的選擇。因此，這就是我們在下一節中將用來對Llama 3.1 8B模型在Google Colab上進行微調的技術。

## 🦙 對 Llama 3.1 8B 進行微調

為了有效地對 [Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) 模型進行微調，我們將使用 Daniel 和 Michael Han 開發的 [Unsloth](https://github.com/unslothai/unsloth) 庫。由於其自定義核心，Unsloth 提供了 2 倍的訓練速度和 60% 的記憶體使用率，相比其他選擇更適合在 Colab 等受限環境中使用。不幸的是，Unsloth 目前只支持單 GPU 設置。對於多 GPU 設置，我推薦流行的替代方案，如 [TRL](https://huggingface.co/docs/trl/en/index) 和 [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)（兩者都將 Unsloth 作為後端）。

在這個例子中，我們將使用QLoRA對[mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)數據集進行微調。這個數據集是[arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)（不包括[arcee-ai/qwen2-72b-magpie-en](https://huggingface.co/datasets/arcee-ai/qwen2-72b-magpie-en)）的子集，我使用[HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)重新過濾了它。請注意，這個分類器不是為了評估指令數據質量而設計的，但我們可以將其用作粗略的代理。結果產生的FineTome是一個超高質量的數據集，包含對話、推理問題、函數調用等。首先，我們需要安裝所有必要的庫。

```
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

安裝完成後，我們可以如下方式匯入它們。

```python
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
```

現在，我們來載入模型。由於我們想要使用QLoRA，我選擇了預先量化的[unsloth/Meta-Llama-3.1-8B-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit)版本。這個4位元精度版本的[meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/blog/mlabonne/meta-llama/Meta-Llama-3.1-8B)比原來的16位元精度模型（16 GB）要小得多（5.4 GB），下載速度也更快。 我們使用bitsandbytes庫以NF4格式載入模型。

在加載模型時，我們必須指定一個最大序列長度，這將限制其上下文視窗。Llama 3.1 支援高達 128k 的上下文長度，但在本例中，我們將其設置為 2,048，因為這將消耗更多的計算資源和 VRAM。最後，dtype 參數會自動檢測您的 GPU 是否支援 BF16 格式，以在訓練期間獲得更好的穩定性（此功能僅限於 Ampere 和更新的 GPU）。

```python
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
```

現在，我們的模型已經以4位元精度加載，我們想要準備它以使用LoRA適配器進行參數高效微調。LoRA有三個重要的參數：
- **Rank**（r），它決定了LoRA矩陣的大小。Rank通常從8開始，但可以增加到256。較高的Rank可以存儲更多的信息，但也會增加LoRA的計算和記憶體成本。這裡我們將其設置為16。

- **Alpha**（α），是一個用於更新的縮放因子。Alpha直接影響了適配器的貢獻，通常設置為1倍或2倍的排名值。

- **Target modules**：LoRA 可以應用於各種模型組件，包括注意力機制（Q、K、V 矩陣）、輸出投影(output projections)、前饋塊(feed-forward blocks)和線性輸出層(linear output layers)。雖然最初著重於注意力機制，但將 LoRA 擴展到其他組件已經顯示出其益處。然而，適應更多模組會增加可訓練參數的數量和記憶體需求。
 
在這裡，我們設定 r=16、α=16，並針對每個線性模組進行最佳化，以達到最佳質量。

為了加快訓練速度，我們不使用 dropout 和偏差（biases）。此外，我們還將使用[Rank-Stabilized LoRA（rsLoRA）](https://arxiv.org/abs/2312.03732)，它修改了LoRA適配器的縮放因子，使其與1/√r成正比，而不是1/r。這使得學習更加穩定（尤其是對於更高的適配器排名(ranks)），並且可以在排名增加時改善微調性能。梯度檢查點由Unsloth處理，以將輸入和輸出嵌入卸載到磁盤並節省VRAM。

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)
```

使用這個LoRA配置，我們只會訓練8億個參數中的4200萬個（0.5196％）。這表明LoRA相比於完整的微調要高效得多。

現在，我們來加載和準備數據集。指令數據集存儲在**特定的格式**中：它可以是Alpaca、ShareGPT、OpenAI等。首先，我們想要解析這種格式以提取指令和答案。[mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)數據集使用ShareGPT格式，具有唯一的"conversations"欄，包含JSONL格式的消息。與Alpaca等簡單格式相比，ShareGPT更適合存儲多輪對話，這更接近用戶與LLM的交互方式。

一旦我們的指令-答案對被解析，我們就想將它們重新格式化以遵循聊天模板。聊天模板是一種用於結構化用戶和模型之間對話的方法。它們通常包括特殊標記來標識消息的開始和結束、誰在說話等。基礎模型沒有聊天模板，所以我們可以選擇任何：ChatML、Llama3、Mistral等。在開源社群中，ChatML 模板（最初來自 OpenAI）是一個流行的選擇。它只是添加兩個特殊標記（`<|im_start|>` 和 `<|im_end|>`）來指示誰在說話。

如果我們將這個模板套用到之前的指令樣本，得到的結果如下：

```
<|im_start|>system
You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.<|im_end|>
<|im_start|>user
Remove the spaces from the following sentence: It prevents users to suspect that there are some hidden products installed on theirs device.
<|im_end|>
<|im_start|>assistant
Itpreventsuserstosuspectthattherearesomehiddenproductsinstalledontheirsdevice.<|im_end|>
```

在下面的程式碼區塊中，我們使用`映射`(`mapping`)參數來解析ShareGPT數據集，並包含ChatML模板。然後，我們加載和處理整個數據集，以將聊天模板應用到每個對話中。

```python
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True)
```

我們現在準備好指定運行的訓練參數。我想簡單介紹一下最重要的超參數：

- **學習率（Learning rate）**：它控制模型更新其參數的強度。學習率太低，訓練會變得緩慢，甚至可能陷入局部最小值。學習率太高，訓練可能會變得不穩定或發散，這會降低模型的性能。
- **學習率調整器（LR scheduler）**：它在訓練過程中調整學習率（LR），初始階段使用較高的學習率以快速取得進展，然後在後期階段逐漸降低學習率。線性調整器（Linear）和餘弦調整器（Cosine）是兩種最常見的選擇。
- **批次大小（Batch size）**：在更新權重之前處理的樣本數量。較大的批次大小通常會導致梯度估計更加穩定，並且可以提高訓練速度，但也需要更多的記憶體。梯度累積（Gradient accumulation）允許有效地使用較大的批次大小，通過在多次前向/後向傳遞之前累積梯度，然後更新模型。
- **訓練週期數（Num epochs）**：訓練數據集的完整遍歷次數。更多的週期使模型能夠多次看到數據，從而可能導致更好的性能。然而，過多的週期可能會導致過擬合。
- **優化器（Optimizer）**：用於調整模型參數以最小化損失函數的演算法。在實踐中，強烈建議使用AdamW 8-bit：它的表現與32-bit版本相同，但使用的GPU記憶體更少。AdamW 的分頁版本（paged version）僅在分散式設定中有意義。
- **權重衰減（Weight decay）**：一種正則化技術，通過在損失函數中添加對大權重的懲罰項，來防止模型過度擬合。它可以鼓勵模型學習更簡單、更普遍的特徵，但是過多的權重衰減可能會阻礙模型的學習。
- **預熱步驟（Warmup steps）**：訓練開始時的一個階段，學習率從一個小值逐漸增加到初始學習率。預熱可以幫助穩定早期訓練，特別是在大學習率或批次大小的情況下，通過允許模型在進行大更新之前調整到數據分佈。
- **打包（Packing）**：批次具有預先定義的序列長度。我們可以將多個小樣本合併為一批，從而提高效率，而不是為每個樣本分配一批。

我使用A100 GPU（40 GB的VRAM）在Google Colab上訓練了整個數據集（100k個樣本）。訓練耗時4小時45分鐘。當然，你可以使用較小的GPU和較小的批次大小，但速度遠遠不及A100。例如，在L4上大約需要19小時40分鐘，而在免費的T4上則需要驚人的47小時。

在這種情況下，我建議只載入數據集的子集，以加快訓練速度。您可以通過修改前面的代碼塊來實現，例如`dataset = load_dataset("mlabonne/FineTome-100k", split="train[:10000]")`，只載入10k個樣本。或者，您也可以使用更便宜的雲端GPU提供商，如Paperspace、RunPod或Lambda Labs。

```python
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()
```

現在模型已經訓練完成，我們用一個簡單的提示來測試它。這不是一個嚴格的評估，而只是快速檢測潛在問題。 我們使用`FastLanguageModel.for_inference()`來獲得2倍的推理速度。

```python
model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
```

模型的回應是 "9.9"，正確！現在，我們來保存訓練好的模型。如果你還記得LoRA和QLoRA的部分，我們訓練的不是模型本身，而是一組適配器（adapters）。在Unsloth中，有三種保存方法：`lora`只保存適配器，`merged_16bit`和`merged_4bit`分別以16位和4位精度將適配器與模型合併。

以下，我們將它們合併為16位精度，以最大化質量。首先，我們將其保存在本地的“model”目錄中，然後上傳到Hugging Face Hub。你可以在[mlabonne/FineLlama-3.1-8B](https://huggingface.co/mlabonne/FineLlama-3.1-8B)找到訓練好的模型。

```python
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("mlabonne/FineLlama-3.1-8B", tokenizer, save_method="merged_16bit")
```

Unsloth還允許您直接將模型轉換為GGUF格式。這是一種為llama.cpp創建的量化格式，與大多數推理引擎兼容，例如[LM Studio](https://lmstudio.ai/)、[Ollama](https://ollama.com/)和oobabooga的[text-generation-webui](https://github.com/oobabooga/text-generation-webui)。由於您可以指定不同的精度（請參閱[GGUF和llama.cpp的文章](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)），我們將迴圈遍歷列表以在`q2_k`、`q3_k_m`、`q4_k_m`、`q5_k_m`、`q6_k`和`q8_0`中量化模型，並將這些量化模型上傳到Hugging Face。[mlabonne/FineLlama-3.1-8B-GGUF](https://huggingface.co/mlabonne/FineLlama-3.1-8B-GGUF)包含所有我們的GGUF模型。

```python
quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
for quant in quant_methods:
    model.push_to_hub_gguf("mlabonne/FineLlama-3.1-8B-GGUF", tokenizer, quant)
```

恭喜！我們從頭開始微調了一個模型，並上傳了您現在可以在您最喜歡的推理引擎中使用的量化模型。請隨意嘗試可用的最終模型[mlabonne/FineLlama-3.1-8B-GGUF](https://huggingface.co/mlabonne/FineLlama-3.1-8B-GGUF)。現在該怎麼做？以下是一些使用您的模型的想法：

- **評估**：在[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)上評估它（你可以免費提交），或使用其他評估工具，如[LLM AutoEval](https://github.com/mlabonne/llm-autoeval)。

- **對齊**：使用偏好數據集（如[mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)）進行直接偏好優化（Direct Preference Optimization），以提升性能。

- **量化**：使用[AutoQuant](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing)將其量化為其他格式，如EXL2、AWQ、GPTQ或HQQ，以實現更快的推理或更低的精度。

- **部屬**：將其部署在Hugging Face Space上，使用[ZeroChat](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC)，適用於已經足夠訓練的模型（約20k個樣本），以跟隨聊天模板。

## 結論

本文提供了監督微調的全面概覽，並示範如何將其應用於Llama 3.1 8B模型。在利用QLoRA的高效記憶體使用情況下，我們成功地對一個8B的LLM進行了微調，儘管只有有限的GPU資源。同時，我們還提供了更高效的替代方案，適用於更大的運行環境，並提出了進一步的建議，包括評估、偏好對齊、量化和部署。希望本指南對您有所幫助。如果您有興趣進一步了解大型語言模型（LLMs），我建議您查看LLM課程。如果您喜歡這篇文章，請在X上關注我@maximelabonne和Hugging Face @mlabonne。祝您微調模型順利！