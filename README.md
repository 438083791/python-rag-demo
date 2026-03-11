# python-rag-demo

Python + LangChain + Ollama + Chroma 实现本地 RAG 问答示例。

## 项目结构

```
python-rag-demo/
├─ testOllamaRAG.py          # 主脚本：加载文档、构建向量库并执行 RAG 问答
├─ docs/
│  └─ 《三国演义》.txt        # 示例知识库文本
├─ chroma_db/               # 运行后生成的 Chroma 向量库（首次运行后出现）
├─ 提示词.ini               # 需求/说明
└─ README.md
```

## 接口说明

当前项目为脚本示例，未提供 HTTP API 接口。

如需通过接口访问，请告知具体的接口设计（路径、方法、参数、返回结构），我可以补充以下内容：
- 接口名称
- 请求参数
- 返回参数
- curl 请求示例
- JSON 返回示例

## 使用方式（命令行）

1. 安装依赖（示例）

```bash
pip install langchain langchain-community langchain-ollama langchain-chroma
```

2. 准备 Ollama 本地模型

- 嵌入模型：`embeddinggemma`
- LLM：`deepseek-r1:1.5b`

3. 运行脚本

```bash
python testOllamaRAG.py
```

## 说明

- `testOllamaRAG.py` 中的 `TextLoader` 默认加载 `./docs/《三国演义》.txt`。
- 首次运行会构建并持久化向量库到 `./chroma_db`。
- 后续可切换为“加载已有向量库”的方式（脚本中已给出注释示例）。
