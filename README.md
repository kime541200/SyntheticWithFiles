<div align="center">
    <img src=./imgs/banner.webp height=40% />
</div>

講述如何使用LLM來產生「基於特定文件」的合成資料集✨️

## Table of content
- [Table of content](#table-of-content)
- [Background](#background)
- [Pre-requirement](#pre-requirement)

## Background
- 合成資料？📃  
  簡單來講就是用生成式AI來產生的資料. (詳見[What is synthetic data?](https://mostly.ai/what-is-synthetic-data))

- 為什麼需要基於特定領域的知識來產生合成資料？🤔  
  1. 在企業內部有許多專業領域知識(domain knowledge)都是只有在該領域的專家才懂, 且這些資料大多都不容易閱讀.
  2. 透過微調讓LLM可以更貼近特定領域的應用場景, 而要微調便需要先準備好資料.

## Pre-requirement
- [Python](https://www.python.org/downloads/release/python-3111/)
- [Ollama](https://ollama.com/download)  
  本文主要作為示範目的, 所以就只用llama3.1-8b-q4_0的模型來跑(效果已經很不錯了🤩)  
  詳見 [How to Run LLM Models Locally with Ollama?](https://www.analyticsvidhya.com/blog/2024/07/local-llm-deployment-with-ollama/)  
    
  (如果想要使用更大的模型, 但是卻沒有足夠的硬體, 非常推薦使用 [Groq](https://groq.com/)🚀 或是 [Nvidia NIM](https://build.nvidia.com/explore/discover)🌲)