# NCCL分析
## 核心步骤
1. 初试化和启动MPI通信。
2. 计算主机名的哈希值，并MPI_allgather通信使得每个rank（进程）都获取其它rank的哈希值。
3. 根据获取的哈希值，比较得到该rank所在的主机参与通信rank本地的localrank序号（哈希值相同的rank在同一主机上，localrank表示该进程在本机的序号）。（哈希值就是主机名，其实可以用主机名来获取主机上参与通信的总rank数，只是主机命名五花八门，哈希值更容易比较）
4. rank0上获取NCCL的唯一ID，并MPI_Bcast广播给其它rank。（这个唯一的ID是用来标识通信组，因此所有通信组中的rank有相同的ID）
5. 基于localrank绑定GPU，并分配发送接收缓冲区，创建CUDA流。
6. 初始化NCCL通信器。
7. nccl allreduce通信。同步CUDA流，确保通信完成。
8. 释放缓冲区。
9. 销毁通信器。
10. 终止MPI环境






## 初始化分析
### NCCL 任务的调度机制
NCCL 采用 异步调度，所有的 NCCL 操作（如 ncclAllReduce、ncclBroadcast、ncclSend、ncclRecv）都不会立即执行，而是：

封装任务到 ncclInfo 结构体。
调用 ncclEnqueueCheck() 将任务放入 NCCL 任务队列。
由 NCCL 统一优化、调度，在 CUDA Stream 上异步执行
### ncclInit()
1. 调用ncclInit()进行nccl库初始化(初步的初始化，基本函数可调用，但还难以通信)
   + 初始化环境，GPU
   + 初始化引导网络，为NCCL网络通信做准备
   + ![init.cc](nccl-master/src/init.cc)
2. 调用bootstrapGetUniqueId()函数来获取一个唯一的ID。
   + 包括两部分：一个随机数+一个环境变量（若无则是bootstrap的网络地址） 
   + ![bootstrap.cc](nccl-master/src/bootstrap.cc)


### 初始化通信器ncclCommInitRank
![init.cc](nccl-master/src/init.cc)
1. 加载CUDA驱动
2. 获取当前CUDA设备ID
3. 根据CUDA设备ID、UniqueId等完成NCCL通信器初始化

![init.cc](nccl-master/src/init.cc)
ncclCommInitRankDev()
1. 检测状态
2. 配置NCCL通信器的一些属性，是否阻塞。通道数量等
   + 对comm-config进行赋值
3. 分配一个作业对象 job,并设置作业对象的各个成员变量
4. 使用 ncclAsyncLaunch 异步启动 ncclCommInitRankFunc 函数来初始化通信

ncclCommInitRankFunc()
1. 获取 CUDA 设备和架构信息，初始化 CUDA 内核
2. 是否有父通信器
    + 有，从父通信器分裂出来子通信器，并初始化
    + 无，直接为其分配内存，并初始化
3. 设置通信器的CUDA架构版本和哈希值。
4. 始化当前通信器的传输层。
5. 加载调整器插件。调整器用于动态调整通信算法，以优化性能。
6. 更新通信器状态为成功，表示通信器初始化成功



### bootstrapInit()
![bootstrap.cc](nccl-master/src/bootstrap.cc)
利用已知的rank0网络地址（UniqueId），建立环形网络，allgather获取所有rank的信息
1. 函数输入ncclUniqueId，从而获得ncclUniqueId中包含的rank0的网络地址，每个rank上都有rank0的网络地址
2. 所有rank根据rank0的网络地址，建立socket并向rank0发送自己的网络地址，rank0上现在就有所有rank的网络地址了
3. rank0告诉每个rank它的下一个节点网络地址，完成**环形网络**建立（方便进行通信）
4. AllGather全局收集所有节点的网络地址

## 代码详细分析
### 集合通信部分
#### all_gather
作用：**所有 GPU 互相收集数据，最终每个 GPU 拥有所有 GPU 的数据。**

先声明NCCL_API宏，确保ncclAllGather和ncclResult_t返回一个NCCL操作的执行结果
sendbuff: 发送缓冲区，包含本地 GPU 的数据。
recvbuff: 接收缓冲区，存放所有 GPU 的数据（所有进程的数据都会聚集到 recvbuff）
comm: NCCL 通信上下文（通信域）。
stream: CUDA 流，在该流上执行 NCCL 操作

[NVTX事件记录](https://gitcode.com/gh_mirrors/nv/NVTX/?utm_source=artical_gitcode&index=top&type=card&&isLogin=1)

该文件中定义了NVTX的数据结构，描述了一个字段
+ 0: 这个是 参数的索引，因为这里只有一个参数，所以索引是 0。
+ NVTX_PAYLOAD_ENTRY_TYPE_SIZE: 这个值表示参数的类型是 数据大小 (size)。
+ "Message size [bytes]": 这个是 参数的名称，用于性能工具中显示

计算消息大小

NVTX事件的记录代码，再Nsight Systems中可视化AllGather操作。**创建一个带有参数的NVTX事件**

构造NCCL操作的信息结构体。目的是封装 ncclAllGather 的参数，并交给 NCCL 调度系统，最终让 ncclEnqueueCheck() 处理该操作
>在 NCCL 代码中，所有的 collective（集体通信）操作都不是直接执行的，而是 通过 ncclInfo 结构体描述任务，然后传递给 ncclEnqueueCheck()，由 NCCL 的调度系统进行排队、优化、执行。

#### all_reduce
作用：**所有 GPU 共享数据并进行归约（如求和、最大值等），最终每个 GPU 拥有相同的归约结果**

参数多了个op，需要指定归约的操作

NVTX额外的归约操作字段，struct NvtxParamsAllReduce，
NVTX 事件的 schema：多了 NVTX_PAYLOAD_ENTRY_NCCL_REDOP字段。记录了op的归约操作，用于在 性能分析工具（如 Nsight Systems）中可视化 归约操作类型。

payload计算的是消息大小+记录

#### broadcast
作用：**根进程（root） 发送数据，所有其他 GPU 接收 root 的数据，最终所有 GPU 都有相同的数据**

NVTX记录数据大小和**root GPU ID**

NVTX事件多了NVTX_PAYLOAD_ENTRY_TYPE_INT字段记录root进程的ID

nccl的参数必须指定，广播需要一个源

#### reduce_scatter
作用：**作用是将所有 GPU 的数据进行归约（如求和、最大值等），然后将结果分散到每个 GPU 上**
同all_reduce

#### reduce
作用：**将多个 GPU 或进程的数据进行归约（如求和、最大值等），并将结果返回到一个指定的根进程（root）**

NVTX记录除了有message siez外，还有root进程的ID和Reduction operation的记录

#### sendrecv
作用：**实现了 NCCL 的点对点通信（Send/Recv）操作，用于在两个 GPU 之间 直接发送和接收数据**

NVTX事件记录中多加了peer的ID

ncclGroupStart(): 开始 NCCL 组操作，避免每次 ncclSend 或 ncclRecv 都单独调度，提高通信效率。
ncclEnqueueCheck(&info): 将 Send 或 Recv 操作加入 NCCL 任务队列。
ncclGroupEnd(): 结束 NCCL 组操作，让所有任务一起执行。
**防止死锁，在组的级别调度操作**















