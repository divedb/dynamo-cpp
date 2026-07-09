Dynamo 的架构分层

从官方架构描述看：

Dynamo
├── Runtime
├── Router
├── Planner
├── KVBM (KV Block Manager)
├── ModelExpress
├── Grove
└── Inference Backend
      ├── vLLM
      ├── TensorRT-LLM
      └── SGLang

其中：

KV-aware Routing → Router
KV Block Manager → KVBM
GPU推理状态 → Backend Worker
Metrics/Statistics → 各组件自身上报

KV Cache 谁维护？

官方直接给出了组件名字：

KV Block Manager (KVBM)

作用：

Offloads KV cache across GPU → CPU → SSD → remote storage

即：

GPU HBM
   ↓
CPU Memory
   ↓
Local SSD
   ↓
Object Storage

KV 的生命周期和位置管理都属于 KVBM。

当前 GPU 上 KV Cache 使用量呢？

这个要分两层看。

Layer 1：实际 KV 数据

真实 KV tensor：

GPU memory

实际保存在：

vLLM
TensorRT-LLM
SGLang

这些 backend 里面。

Layer 2：KV 元数据

Dynamo 维护：

哪个 worker 有哪些 block
哪些 block 热
哪些 block 被 pin
哪些 block 被 offload

这些集群级信息。

官方描述：

tracks KV cache across large fleets of GPUs

stores KV locations

cache overlap

说明 Dynamo 有一个全局 KV 索引。

所以：

实际KV
    ↓
backend维护

KV位置索引
    ↓
Dynamo KVBM维护
Statistics 谁维护？

Dynamo 是微服务架构。

统计通常是组件各自维护。

例如：

Router

维护：

cache hit rate
routing latency
worker load
overlap score

KVBM

维护：

gpu cache blocks
cpu cache blocks
ssd cache blocks

eviction count
offload count
pin count

Backend Worker

维护：

active requests
batch size
tokens/s
TTFT
ITL
GPU memory
KV memory

这些实际上来自：

vLLM metrics
TRT-LLM metrics
SGLang metrics

然后被 Dynamo 聚合。