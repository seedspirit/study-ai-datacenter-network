# AI/ML 인프라를 위한 네트워크 기술 입문

AI/ML 인프라의 네트워크는 단순한 API 서버와 DB 사이의 통신 문제가 아님. 대규모 학습과 LLM 추론에서는 GPU가 계산한 **Tensor, Gradient, Activation, KV Cache**를 다른 GPU로 계속 옮겨야 함.

네트워크가 느리면 GPU는 계산을 끝내고도 다음 데이터를 기다리게 됨. 결국 AI/ML 네트워크의 핵심 질문은 다음과 같음.

> **GPU Memory의 데이터를 얼마나 빠르고 안정적이며 예측 가능하게 다른 GPU Memory 근처까지 옮길 수 있는가?**

이 관점에서 보면 NVIDIA의 기술은 하나의 흐름으로 이어짐.

```text
PCIe → NVLink → NVSwitch → GPUDirect RDMA → InfiniBand → NCCL
```

---

## 1. 먼저 알아야 할 기초 용어

### Interconnect

말 그대로 **서로 연결해 주는 통신 경로**. AI/ML 인프라에서는 CPU, GPU, NIC, 서버 노드, 스위치를 서로 연결하는 고속 통신 기술을 의미함. 무엇과 무엇을 연결하는지에 초점을 둔 개념이라고 볼 수 있음

| 연결 대상 | 주요 기술 |
| --- | --- |
| CPU ↔ GPU | PCIe |
| GPU ↔ GPU | PCIe, NVLink |
| 여러 GPU ↔ 여러 GPU | NVSwitch |
| 서버 ↔ 서버 | InfiniBand, Ethernet/RoCE |
| GPU ↔ 원격 GPU | GPUDirect RDMA + InfiniBand/RoCE |


### Fabric

여러 Interconnect, NIC, 스위치, 케이블, 라우팅, 혼잡 제어가 엮여 하나의 통신망처럼 동작하는 구조. 직물이 실로 엮이듯 데이터센터의 서버와 스위치가 촘촘히 연결된 형태.

- **NVSwitch Fabric:** 서버 내부의 여러 GPU를 NVLink 기반으로 연결
- **InfiniBand Fabric:** 여러 GPU 서버를 고성능 네트워크로 연결
- **Ethernet Fabric:** Ethernet Switch 기반의 데이터센터 네트워크
- **AI Fabric:** GPU 학습·추론 트래픽을 처리하는 전체 네트워크 구조

> **Interconnect** = 연결 기술<br>
> **Fabric** = 연결 기술이 모여 만든 전체 통신 구조

### Topology

**무엇이 무엇과 어떤 경로로 연결되어 있는지**를 나타내는 구조. AI 서버의 성능에 직접적인 영향을 줌.

GPU 0과 GPU 1이 NVLink로 직접 연결된 경우:

```text
GPU 0 ── NVLink ── GPU 1
```

PCIe와 CPU Socket을 거쳐야 하는 경우:

```text
GPU 0 ── PCIe ── CPU ── PCIe ── GPU 1
```

단순한 GPU 개수보다 GPU끼리 어떤 경로로 연결되어 있는지가 중요

### Bandwidth와 Latency

| 구분 | 의미 | 중요한 상황 |
| --- | --- | --- |
| **Bandwidth** | 초당 전송 가능한 데이터 양 | 큰 Gradient, Activation, KV Cache 이동 |
| **Latency** | 전송 시작부터 수신까지 걸리는 시간 | MoE Dispatch, Decode 동기화, 세분화된 Collective |

- 큰 Tensor를 한 번에 전송 → Bandwidth 중요
- 작은 데이터를 자주 교환 → Latency 중요
- 대규모 학습·추론 → 둘 다 중요

---

## 2. PCIe: GPU와 NIC가 시스템에 붙는 기본 통로

PCIe는 GPU, NIC, NVMe SSD 같은 장치를 CPU와 메인보드에 연결하는 기본 I/O Bus. NVIDIA GPU도 기본적으로 PCIe를 통해 시스템에 연결됨.

```text
CPU
 └─ PCIe
     ├─ GPU
     ├─ NIC
     └─ NVMe SSD
```

범용성은 높지만 GPU 간 통신만 놓고 보면 항상 최적의 경로는 아님. NVLink가 없다면 GPU 0에서 GPU 1로 데이터를 보낼 때 PCIe Root Complex나 CPU 경로를 거칠 수 있음.

### 해결하는 문제

- GPU, NIC, NVMe 같은 고속 장치를 CPU와 연결
- GPU와 NIC가 외부 네트워크로 나가기 위한 기본 경로 제공
- PCIe 기능을 활용해 GPUDirect RDMA의 GPU-NIC 직접 경로 구성

---

## 3. NVLink: GPU끼리 직접 연결하는 고속 링크

NVIDIA GPU 사이의 고속 Interconnect. PCIe 기반 Multi-GPU 통신의 한계를 줄이고 GPU끼리 빠르게 직접 통신할 수 있는 전용 경로 제공.

```text
# PCIe 기반 GPU-GPU 통신
GPU 0 ── PCIe/CPU 경로 ── GPU 1

# NVLink 기반 GPU-GPU 통신
GPU 0 ── NVLink ── GPU 1
```

NVIDIA 설명 기준 H100의 4세대 NVLink는 GPU-to-GPU Interconnect로 900 GB/s 대역폭 제공. GPU 간 Tensor 이동이 많은 모델 병렬화에서 중요한 특성. NVIDIA NVLink Switch는 NVLink 연결을 노드 간으로 확장해 고대역폭 Multi-Node GPU Cluster 구성 가능.[^1]

### 해결하는 문제

- GPU-GPU 데이터 이동에서 PCIe 병목 완화
- Tensor Parallelism의 Layer 중간 결과 교환 비용 감소
- Pipeline Parallelism의 Activation 전달 비용 감소
- Single-Node Multi-GPU NCCL Collective 성능 개선

NVLink가 있어도 GPU VRAM이 자동으로 하나의 거대한 Memory처럼 합쳐지는 것은 아님. Framework와 Runtime에서 Multi-GPU 통신을 명시적으로 사용해야 함

---

## 4. NVSwitch: 여러 GPU를 하나의 GPU Fabric으로 묶기

GPU가 2장이라면 NVLink를 이용한 직접 연결 가능. 하지만 8장 이상이면 모든 GPU 쌍을 균일하게 빠른 경로로 연결하기 어려움. 이 문제를 해결하기 위한 Switch Chip이 **NVSwitch**.

```text
GPU 0 ┐
GPU 1 ┤
GPU 2 ┤
GPU 3 ┤── NVSwitch Fabric ── GPU 4
GPU 5 ┤
GPU 6 ┤
GPU 7 ┘
```

NVIDIA 문서는 NVSwitch를 여러 GPU를 하나의 시스템 안에서 연결하는 High-Bandwidth, Low-Latency Fabric으로 설명.[^2]

### 해결하는 문제

- GPU 증가 시 특정 GPU 쌍만 빠르고 나머지는 느려지는 문제 완화
- 8-GPU 서버에서 균일한 GPU-GPU 통신 경로 제공
- Tensor Parallelism, MoE All-to-All, NCCL All-Reduce 등 Collective Communication 최적화
- 여러 GPU를 하나의 큰 가속기처럼 활용하기 위한 내부 Fabric 제공

> **NVLink** = GPU 사이의 고속 링크<br>
> **NVSwitch** = 여러 NVLink를 엮어 GPU 간 통신을 중계하는 스위치

---

## 5. DMA와 RDMA: CPU를 덜 거치는 데이터 이동

### DMA

**Direct Memory Access**. 장치가 CPU 대신 Memory에 직접 읽고 쓰는 방식.

```text
NIC / GPU / NVMe ── DMA ── Memory
```

### RDMA

**Remote Direct Memory Access**. 네트워크 너머 다른 서버의 Memory에 직접 접근하는 기술.

```text
Server A Memory ── NIC ── Network ── NIC ── Server B Memory
```

일반 TCP/IP 통신은 Kernel Network Stack, Socket Buffer, Memory Copy, CPU Interrupt 등을 거침. RDMA는 데이터 경로에서 CPU와 OS 개입을 줄여 낮은 Latency, 낮은 CPU Load, 높은 Bandwidth를 목표로 함.

NVIDIA RDMA 문서에서도 한 Host의 Memory에서 다른 Host의 Memory로 Remote OS와 CPU 개입 없이 직접 접근할 수 있는 방식으로 설명.[^3]

### 해결하는 문제

- Multi-Node 학습의 Gradient Synchronization 비용 감소
- 서버 간 Tensor 이동 시 CPU와 Kernel Stack 병목 완화
- 대규모 GPU Cluster의 통신 Overhead 감소

---

## 6. GPUDirect RDMA: GPU Memory와 NIC를 직접 연결

AI/ML 데이터는 주로 CPU Memory가 아닌 GPU Memory에 위치. Gradient, Activation, KV Cache 모두 GPU Memory 위에 존재. 이 데이터를 네트워크로 보낼 때마다 CPU Memory를 거치면 비용 증가.

### 일반 경로

```text
GPU Memory
    ↓
CPU Memory
    ↓
   NIC
    ↓
 Network
    ↓
   NIC
    ↓
CPU Memory
    ↓
GPU Memory
```

### GPUDirect RDMA 경로

```text
GPU Memory
    ↓
   NIC
    ↓
 Network
    ↓
   NIC
    ↓
GPU Memory
```

NVIDIA CUDA 문서에서는 GPU와 Network Interface, Storage Adapter 같은 Third-Party Peer Device 사이에 PCI Express 기능을 이용한 직접 데이터 교환 경로를 제공하는 기술로 설명.[^4]

### 해결하는 문제

- GPU 데이터를 서버 밖으로 보낼 때 CPU Memory 경유 비용 감소
- Multi-Node NCCL 통신 성능 개선
- 분산 학습에서 GPU의 통신 대기 시간 감소
- Disaggregated Inference의 KV Cache 이동 비용 감소

---

## 7. InfiniBand: AI/HPC용 고성능 네트워크 Fabric

HPC와 AI Cluster에서 많이 사용하는 고성능 Interconnect. 일반 Ethernet/TCP가 범용 네트워크라면 InfiniBand는 낮은 Latency, 낮은 CPU Overhead, 높은 Bandwidth, RDMA를 중심으로 설계된 Cluster Network.

NVIDIA 문서는 InfiniBand를 High-Speed, Low-Latency, Low CPU Overhead, Scalable Server/Storage Interconnect로 설명. 핵심 기능 중 하나는 Native RDMA 지원.[^5]

### 주요 구성 요소

| 구성 요소 | 역할 |
| --- | --- |
| **HCA** | InfiniBand용 NIC |
| **InfiniBand Switch** | 서버 사이 트래픽 중계 |
| **Cable / Transceiver** | 물리적 연결 및 신호 전송 |
| **Subnet Manager** | Fabric 구성 및 경로 관리 |
| **RDMA Verbs** | RDMA 통신을 위한 API |
| **QoS / Congestion Control** | 품질 및 혼잡 관리 |

### 해결하는 문제

- 여러 GPU 서버를 하나의 학습 Cluster로 연결
- Multi-Node All-Reduce, Reduce-Scatter, All-Gather, All-to-All 가속
- 대규모 GPU Cluster에서 예측 가능한 통신 성능 제공
- CPU의 과도한 네트워크 처리 부담 완화

> **NVLink / NVSwitch** = 서버 내부 GPU Fabric<br>
> **InfiniBand** = 서버 간 GPU Cluster Fabric

---

## 8. NCCL: GPU Collective Communication의 중심

NVIDIA GPU와 Networking에 최적화된 Collective Communication Library. Multi-GPU, Multi-Node Communication Primitive 구현.[^6]

### 대표 연산

- All-Reduce
- All-Gather
- Reduce-Scatter
- Broadcast
- All-to-All
- Send / Receive

PyTorch에서는 다음과 같이 호출.

```python
torch.distributed.all_reduce(tensor)
```

내부적으로 동작하는 계층:

```text
PyTorch
   ↓
torch.distributed
   ↓
NCCL
   ↓
NVLink / NVSwitch / PCIe / GPUDirect RDMA
   ↓
InfiniBand 또는 Ethernet Fabric
```

NCCL을 사용하면 GPU 간 통신 알고리즘을 직접 구현할 필요 없음. Topology와 Transport에 맞는 Collective Communication 수행 가능.

---

## 9. 학습과 추론에서 해결하는 문제

### 학습

여러 GPU가 각자 계산한 결과를 지속적으로 동기화해야 함.

#### Data Parallel

- 각 GPU가 서로 다른 Batch 계산
- Gradient를 All-Reduce로 통합
- 주요 기술: **NCCL, NVLink, NVSwitch, InfiniBand, GPUDirect RDMA**

#### Tensor Parallel

- 하나의 Layer 계산을 여러 GPU에 분산
- Layer마다 All-Gather, Reduce-Scatter 발생
- 주요 기술: **NVLink, NVSwitch, NCCL**

#### MoE

- Token을 Expert가 있는 GPU로 Dispatch
- 계산 후 다시 Combine
- 주요 기술: **NVSwitch, InfiniBand, All-to-All Communication**

### 추론

모델이 커질수록 LLM 추론에서도 통신이 주요 문제로 부상.

#### Tensor Parallel Inference

- 큰 모델을 여러 GPU에 분산 배치
- Decode 단계에서도 GPU 간 통신 발생
- 주요 기술: **NVLink, NVSwitch, NCCL**

#### Disaggregated Serving

- Prefill Worker와 Decode Worker 분리
- Worker 사이에서 KV Cache 이동
- 주요 기술: **GPUDirect RDMA, InfiniBand/RoCE**

#### Multi-Node Serving

- 한 서버에 모델 전체를 올릴 수 없거나 Throughput을 높여야 할 때 사용
- 주요 기술: **InfiniBand, GPUDirect RDMA, NCCL**

학습과 추론의 핵심은 동일함. **GPU가 계산한 데이터를 다음 계산 위치로 빠르게 전달해야 함.**

| 구간 | 담당 기술 |
| --- | --- |
| 서버 내부 | NVLink, NVSwitch |
| 서버 간 | GPUDirect RDMA, InfiniBand |
| 통신 연산 | NCCL |

---

## 마무리

AI/ML 네트워크 기술을 단순히 서버 간 연결로만 보면 이해하기 어려움. 핵심은 GPU Memory 사이의 데이터 이동

- **PCIe:** GPU와 NIC가 시스템에 붙는 기본 통로
- **NVLink:** GPU끼리 직접 연결하는 고속 링크
- **NVSwitch:** 여러 GPU를 하나의 내부 Fabric으로 연결
- **GPUDirect RDMA:** GPU Memory와 NIC 사이의 직접 경로 제공
- **InfiniBand:** 여러 서버를 고성능 RDMA Fabric으로 연결
- **NCCL:** 하드웨어 경로 위에서 GPU Collective Communication 수행

NVIDIA 네트워크 기술 Stack이 향하는 목표는 GPU가 계산보다 통신을 기다리는 시간 최소화하는 것으로 볼 수 있음


