NCCL과 GPU 인프라: 여러 GPU는 어떻게 하나의 시스템처럼 일하는가

AI/ML 인프라에서 GPU 성능은 단순히 “GPU 한 장이 얼마나 빠른가”로만 결정되지 않는다. 모델이 커질수록 하나의 GPU에 모든 weight, activation, gradient, KV cache를 올리기 어렵고, 결국 여러 GPU가 하나의 작업을 나눠 수행해야 한다. 이때 GPU들이 계산한 중간 결과를 서로 주고받는 비용이 전체 성능을 좌우한다.

NCCL은 바로 이 지점에 있는 NVIDIA의 collective communication library다. NVIDIA 공식 문서는 NCCL을 topology-aware inter-GPU communication primitive library로 설명한다. 즉 NCCL은 여러 GPU가 함께 계산할 때 필요한 데이터 교환을 GPU 연결 구조와 네트워크 topology에 맞춰 효율적으로 처리해주는 계층이다.

⸻

1. NCCL은 GPU 간 collective communication을 어떻게 맡는가

NCCL은 여러 GPU가 함께 작업할 때 필요한 collective communication을 제공한다. NVIDIA NCCL 페이지는 NCCL이 all-gather, all-reduce, broadcast, reduce, reduce-scatter, point-to-point send/receive 같은 routine을 제공한다고 설명한다.

대표적인 collective는 다음과 같다.

* all-reduce
    * 여러 GPU의 값을 합치고, 결과를 모든 GPU가 갖는다.
    * data parallel training의 gradient synchronization에서 자주 사용된다.

GPU0 gradient ┐
GPU1 gradient ├─ all-reduce ─ 모든 GPU가 같은 gradient 합계를 가짐
GPU2 gradient │
GPU3 gradient ┘

* all-gather
    * 각 GPU가 가진 shard를 모아 모든 GPU가 전체 데이터를 갖게 한다.
    * tensor parallelism에서 자주 등장한다.
* reduce-scatter
    * 여러 GPU의 데이터를 reduce한 뒤, 결과를 GPU별 shard로 나눠 가진다.
    * ZeRO, tensor parallelism, distributed optimizer에서 중요하다.
* all-to-all
    * 각 GPU가 다른 모든 GPU와 서로 다른 데이터를 교환한다.
    * MoE의 expert dispatch/combine에서 중요하다.

PyTorch에서는 사용자가 보통 다음과 같이 호출한다.

torch.distributed.all_reduce(tensor)

하지만 backend가 NCCL이면 내부에서는 대략 다음 계층이 움직인다.

PyTorch
  ↓
torch.distributed
  ↓
NCCL
  ↓
CUDA kernel / GPU memory operation
  ↓
NVLink / NVSwitch / PCIe / InfiniBand / RDMA

NCCL이 중요한 이유는 단순히 “데이터를 보내준다”가 아니다. GPU들이 어떤 경로로 연결되어 있는지에 따라 통신 비용이 크게 달라지기 때문이다.

예를 들어 다음 경로들은 모두 성능이 다르다.

GPU0 ─ NVLink ─ GPU1          빠름
GPU0 ─ PCIe ─ CPU ─ GPU2       상대적으로 느릴 수 있음
GPU0 ─ NIC ─ Network ─ GPU7    원격 노드

NCCL은 이런 topology를 고려해 통신 경로와 알고리즘을 선택한다. NVIDIA의 NCCL 관련 자료도 NCCL이 PCIe, NVLink, Ethernet, InfiniBand interconnect 위에서 high bandwidth와 low latency를 달성하도록 최적화되어 있다고 설명한다.

또한 NCCL은 collective operation을 수행하기 위해 여러 알고리즘을 사용한다. NVIDIA의 NCCL deep dive 글은 NCCL이 Ring, Tree, PAT 계열 알고리즘을 포함한다고 설명한다.

* Ring
    * GPU들을 고리처럼 연결해 데이터를 순차적으로 돌리는 방식
    * 큰 메시지에서 bandwidth 활용이 좋음

GPU0 → GPU1 → GPU2 → GPU3
 ↑                    ↓
 └────────────────────┘

* Tree
    * 계층 구조로 데이터를 모으고 다시 퍼뜨리는 방식
    * 작은 메시지나 latency-sensitive communication에 유리할 수 있음
* PAT
    * Parallel Aggregated Trees 계열 알고리즘
    * 여러 tree를 병렬적으로 활용해 collective 성능을 개선하는 방향으로 이해하면 충분함

처음부터 세부 구현을 파고들 필요는 없다. 중요한 것은 NCCL이 collective 종류, 메시지 크기, GPU topology, 네트워크 topology에 따라 다른 통신 전략을 선택한다는 점이다.

⸻

2. 학습과 추론에서 NCCL이 해결하는 병목

NCCL은 학습에서 특히 중요하다. 학습은 매 step마다 계산과 통신이 반복되기 때문이다.

학습

Data Parallel에서는 각 GPU가 서로 다른 mini-batch를 처리한 뒤 gradient를 합쳐야 한다.

* 주요 collective: all-reduce
* 병목: gradient synchronization
* 관련 기술: NCCL, NVLink, NVSwitch, InfiniBand, GPUDirect RDMA

GPU 수가 늘어날수록 계산량은 분산되지만, gradient synchronization 비용도 커진다. 따라서 대규모 학습 성능은 GPU FLOPS만이 아니라 NCCL all-reduce 성능에도 크게 의존한다.

Tensor Parallel에서는 하나의 layer를 여러 GPU에 나눠 계산한다. 예를 들어 거대한 matrix multiplication을 여러 GPU가 shard 단위로 수행한다.

* 주요 collective: all-gather, reduce-scatter
* 병목: layer마다 발생하는 tensor shard 교환
* 관련 기술: NCCL, NVLink, NVSwitch

Tensor parallelism은 모델을 더 큰 GPU memory 공간에 나눠 올릴 수 있게 해주지만, 그 대가로 layer마다 통신이 발생한다. 따라서 NVLink/NVSwitch가 있는 서버와 PCIe만 있는 서버는 성능 차이가 날 수 있다.

MoE에서는 token마다 어떤 expert로 보낼지 router가 결정한다. expert가 다른 GPU에 있으면 token representation을 네트워크로 보내야 한다.

* 주요 collective: all-to-all
* 병목: expert dispatch/combine
* 관련 기술: NCCL, NVSwitch, InfiniBand

MoE는 계산량을 효율적으로 분산할 수 있지만, 통신 패턴이 복잡해진다. 특히 all-to-all은 단순 all-reduce와 다른 병목을 만든다.

추론

작은 모델을 GPU 한 장에 올려 serving한다면 NCCL의 중요성이 상대적으로 작을 수 있다. 하지만 큰 LLM을 여러 GPU에 나눠 serving하는 순간 추론에서도 통신이 중요해진다.

Tensor Parallel Inference에서는 모델의 layer가 여러 GPU에 나뉘어 올라간다. decode 단계에서도 GPU 간 통신이 반복되므로, batch가 작으면 latency가 중요하고 batch가 크면 bandwidth도 중요하다.

Pipeline Parallel Inference에서는 layer를 GPU별 stage로 나눈다.

GPU0: layer 1~10
  ↓ activation
GPU1: layer 11~20
  ↓ activation
GPU2: layer 21~30

이때 인접 stage가 빠른 interconnect로 연결되어 있는지가 latency에 영향을 준다.

Disaggregated Serving에서는 prefill worker와 decode worker를 분리할 수 있다. prefill은 긴 prompt를 처리해 KV cache를 만들고, decode는 이를 이어받아 token을 생성한다. 이 경우 KV cache transfer가 중요해지며, 서버 간 이동이라면 GPUDirect RDMA와 InfiniBand/RoCE 성능이 중요해진다. NVIDIA CUDA 문서는 GPUDirect RDMA가 GPU와 네트워크 인터페이스 같은 third-party peer device 사이에 직접 데이터 교환 경로를 제공한다고 설명한다.

결국 학습과 추론 모두에서 NCCL은 GPU들이 “따로 계산하는 장치들”이 아니라 “함께 계산하는 하나의 병렬 시스템”처럼 동작하도록 만든다. GPU 아키텍처가 빠른 tensor 계산을 담당한다면, NCCL은 여러 GPU가 그 계산 결과를 빠르게 공유하도록 만드는 통신 계층이다.

⸻

3. 인프라 관점에서 NCCL을 어떻게 봐야 하나

인프라 엔지니어에게 NCCL은 단순한 ML 라이브러리가 아니다. 실제로는 GPU 서버의 topology, NIC 배치, Kubernetes 네트워크, 컨테이너 런타임, 드라이버, CUDA, RDMA 설정이 모두 맞물려야 제대로 성능이 나오는 분산 시스템의 데이터 경로에 가깝다.

먼저 봐야 할 것은 topology다.

nvidia-smi topo -m

이 명령으로 다음을 확인할 수 있다.

* GPU끼리 NVLink로 연결되어 있는가?
* GPU 간 통신이 PCIe만 타는가?
* GPU와 NIC가 가까운가?
* NUMA node를 건너야 하는가?
* 특정 GPU만 네트워크 통신에 불리한 위치에 있는가?

NCCL 성능 문제는 종종 “GPU가 느리다”가 아니라 “GPU가 잘못된 경로로 통신한다”에 가깝다. 예를 들어 multi-node training에서 GPU와 NIC가 먼 NUMA 위치에 있거나, NCCL이 의도하지 않은 network interface를 선택하면 성능이 크게 떨어질 수 있다.

다음으로 중요한 것은 NCCL 로그다.

NCCL_DEBUG=INFO

NCCL 로그를 보면 어떤 transport, interface, algorithm이 선택되었는지 파악할 수 있다. 운영 환경에서는 다음과 같은 설정을 확인하게 된다.

* NCCL_SOCKET_IFNAME
    * NCCL이 사용할 network interface 지정
* NCCL_IB_HCA
    * 사용할 InfiniBand/RDMA HCA 지정
* NCCL_DEBUG
    * NCCL 디버그 로그 활성화
* CUDA_VISIBLE_DEVICES
    * 프로세스가 볼 GPU 순서와 범위 제어
* NVIDIA_VISIBLE_DEVICES
    * 컨테이너에서 노출할 GPU 제어

Kubernetes 환경에서는 문제가 더 복잡해진다. 일반적인 Pod-to-Pod 네트워크가 잘 된다고 해서 NCCL 통신이 최적인 것은 아니다.

일반 서비스 트래픽은 대략 다음 경로를 탄다.

Pod → CNI → Service → Pod

하지만 고성능 GPU 통신은 다음에 가깝다.

Container process
  ↓
NCCL
  ↓
CUDA / RDMA library
  ↓
GPU memory / NIC
  ↓
InfiniBand or RoCE fabric

따라서 AI 클러스터 운영에서는 다음 요소를 함께 봐야 한다.

* NVIDIA Driver / CUDA / NCCL 버전 호환성
* GPU Operator를 통한 GPU device 노출
* RDMA device plugin 또는 Network Operator 구성
* hostNetwork, Multus, SR-IOV 사용 여부
* Pod 안에서 /dev/infiniband 장치가 보이는지
* 컨테이너 안에서 NCCL이 올바른 NIC를 선택하는지
* MIG 사용 시 NCCL collective와 topology 제약이 있는지
* 노드 간 MTU, PFC/ECN, RoCE 설정이 맞는지
* NCCL test로 실제 all-reduce 성능이 나오는지

실제로는 학습 코드보다 인프라 설정이 병목인 경우도 많다. PyTorch 코드는 동일해도, GPU topology, NVLink 활성화 여부, RDMA 설정, 컨테이너 device mount, network interface 선택에 따라 throughput이 크게 달라질 수 있다.

그래서 운영자는 모델 코드만 보는 것이 아니라, 다음 계층을 함께 봐야 한다.

Application
  ↓
PyTorch / vLLM / DeepSpeed
  ↓
torch.distributed / NCCL
  ↓
CUDA / GPU Driver
  ↓
NVLink / PCIe / GPUDirect RDMA
  ↓
InfiniBand / RoCE / Ethernet
  ↓
Switch / NIC / Cable / Topology

NCCL을 잘 이해한다는 것은 단순히 all-reduce의 의미를 아는 것이 아니다. GPU 서버에서 실제 데이터가 어떤 경로로 흐르고, 그 경로가 왜 느려질 수 있으며, 어떤 계층에서 병목을 확인해야 하는지 아는 것이다.

인프라 관점에서 NCCL은 “ML 코드의 내부 구현”이라기보다, GPU 클러스터의 성능을 드러내는 관측 지점이다. NCCL benchmark가 잘 나오지 않으면 학습과 추론도 잘 나오기 어렵다. 반대로 NCCL 경로가 잘 잡혀 있으면, 그 위의 PyTorch, DeepSpeed, vLLM 같은 프레임워크가 multi-GPU 자원을 더 안정적으로 활용할 수 있다.
