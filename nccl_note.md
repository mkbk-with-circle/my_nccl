[toc]
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

### 初始化分析
#### bootstrap.cc
[bootstrap.cc](nccl/src/bootstrap.cc)
进程间通信的初始化机制
+ 进程发现（Process Discovery）
+ 通信参数交换（Exchange Communication Parameters）
+ 建立初始连接（Establish Initial Connections）





bootstrapInit():初始化引导的**网络接口**
bootstrapNetSend(sock, data, size) / bootstrapNetRecv(sock, data, size)：通过给定的套接字 (ncclSocket) 发送 / 接收数据。
setFilesLimit()：通过 getrlimit / setrlimit 将进程可打开文件句柄数（RLIMIT_NOFILE）设置为系统最大值。
bootstrapRoot(void* rargs)在 “Root” 线程中运行，负责接收所有 rank（进程）的连接信息并将其分发给各个 rank
+ 依次向每个rank发送“下一个节点”的地址，用于构建ring拓扑环

bootstrapCreateRoot(handle, idFromEnv)：创建 Bootstrap “Root” 的上下文，开启一个新线程跑 bootstrapRoot
bootstrapGetUniqueId(handle)：生成一个全局唯一的 NCCL 引导 ID（即 ncclBootstrapHandle），里面含有随机 magic 和要监听的地址。
+ 包括两部分：一个随机数+一个环境变量（若无则是bootstrap的网络地址）

bootstrapInit(handle, comm)：在非 Root 进程中，基于 handle 与 comm（NCCL 通信上下文）进行引导初始化，完成本 rank 与 root 及其他 rank 的地址交换
+ 通过ncclSocketConnect 把自己的 extInfo（含地址）发给 root，让 root 知道我是谁及如何连我
+ 从 root 等待 ring 上 “下一跳” rank 的地址，然后连接（ringSendSocket），再 accept 上一个 rank 连接（ringRecvSocket）。
+ 进行 AllGather，收集所有 rank 的监听地址。之后还创建 proxy 相关的监听端口。

bootstrapAllGather(commState, allData, size)：把每个 rank 的数据都发到所有 rank
+ n-1次迭代，反复接受上一片数据并且发给下一片

bootstrapSend(commState, peer, tag, data, size)：发送数据给 peer
bootstrapBarrier(commState, ranks, rank, nranks, tag)：等待所有 rank 完成，在一组（ranks）之内实现 barrier 同步，
bootstrapIntraNodeAllGather：只在给定的进程组（ranks）内做 AllGather，跟 bootstrapAllGather 类似，但只在同节点或更小范围内使用
bootstrapIntraNodeBroadcast：只在给定的进程组（ranks）内做 Broadcast，同上
bootstrapRecv(commState, peer, tag, data, size)：在引导通信中，从指定 peer + tag 接收数据
+ 先看“unexpected”队列，看有没有意外的连接
+ 若无，则阻塞Accept新的连接，判断发来的新消息是否符合(peer, tag)，若符合，则接收数据，否则保存到“unexpected”队列

#### channel.cc
[channel.cc](nccl/src/channel.cc)
initChannel(struct ncclComm* comm, int channelId)
+ 初始化通道，为每个通道创建一个ncclChannel结构体，并初始化
+ 为通道分配内存，包括 peer 连接信息、环形拓扑（Ring）信息，并且初始化peers结构

#### debug.cc
[debug.cc](nccl/src/debug.cc)
ncclDebugInit()：初始化 NCCL 日志系统（设置日志级别、日志文件、调试子系统）
ncclDebugLog(level, flags, filefunc, line, fmt, …)	：通用日志打印函数，供 INFO, WARN, TRACE 级别日志使用
ncclSetThreadName(thread, fmt, …)：设置 NCCL 线程名称，帮助调试


#### net.cc
[net.cc](nccl/src/net.cc)
提供不同版本的NCCL网络接口


#### enqueue.cc
[enqueue.cc](nccl/src/enqueue.cc)
1. 配置了不同的核，最后可以根据不同算法，不同协议，不同归约操作及不同数据类型自动注册通信kernel组合。
2. ncclInitKernelsForDevice：获取核函数的 最大栈大小 (maxStackSize)。设定 共享内存 carveout 以及 最大动态共享内存大小

##### Launch system : synchronization and CUDA kernel launch 
1. appendWorkElemColl() 和 appendWorkElemP2p() 向 NCCL 的计划 (ncclKernelPlan) 中添加 Collective 和 P2P 任务
2. addCollToPlan() 选择 负载最小的通信通道 并将 collective 任务分配到这些通道。
3. addP2pToPlan() 负责 P2P 任务的调度，选择最优的通信协议 (LL 或 SIMPLE)，并设置 Proxy 任务。
4. scheduleCollTasksToPlan（）：负责将 NCCL Collective 任务调度到 ncclKernelPlan 里，并优化任务执行
   + 任务聚合
   + 计算任务调度信息 
5. scheduleP2pTasksToPlan：负责 Point-to-Point (P2P) 任务的调度，用于 GPU 之间直接通信
6. uoloadWork：将 ncclKernelPlan 里的任务真正提交到 GPU，以便 CUDA Kernel 可以执行
7. uploadProxyOps：处理 代理任务 (Proxy Operations)，确保 跨节点 (Inter-Node) 通信可以顺利进行。


执行的函数
1. ncclLaunchPrepare（）：准备并调度 NCCL 任务计划 (ncclKernelPlan)
   + 确定 NCCL 任务如何执行
   + 是否 批量执行 (Batch Execution)
   + 是否 持久化 (Persistent Execution)
2. ncclLaunchKernel（）：执行 NCCL 任务计划 (ncclKernelPlan)

##### Enqueueing system : computation of kernel and proxy operations parameters
1. getCollNetSupport()：获取 NCCL 通信支持的网络类型，是否支持CollNet
2. getAlgoInfo()：获取 NCCL 通信支持的算法信息，选择最短时间的算法
3. getPatternInfo()：根据 NCCL_ALGO_* 算法，选择 最佳通信模式 (Pattern)
4. getLoopInfo()：计算任务执行时 每个 Loop 需要执行的步数 (Steps)
5. hostToDeviceColl()：将 Host 端的 Reduction Op 转换为 GPU 端可执行的 Reduction Op
6. taskAppend()
   + 区分 P2P (Send/Recv) 和 Collective (AllReduce, Reduce, Broadcast)，并且加入队列中
   + 将 info 结构体转换为 NCCL 任务 (ncclTaskP2p 或 ncclTaskColl)
7. ncclEnqueueCheck：检查参数合法性，确保通道可用，最后把任务加入NCCL调度队列

执行的函数
1. computeColl：
   + 计算 NCCL Collective (如 AllReduce, Broadcast, Reduce 等) 操作的最佳执行参数
   + 分配计算资源，包括通信拓扑、线程、chunkSize、数据切片策略


#### init_nvtx.cc
[init_nvtx.cc](nccl/src/init_nvtx.cc)
在 NVTX（NVIDIA Tools Extension）中注册 NCCL 的 Reduction 操作类型，使得在使用 NCCL 进行 GPU 通信时，可以在 NVTX 事件追踪中更直观地查看 NCCL Reduction 操作的信息
将原本的数值映射到具体的操作

#### group.cc
[group.cc](nccl/src/group.cc)
NCCL 的组操作管理
核心结构
+ ncclAsyncJob：定义了一个异步任务的通用结构，包括函数指针、destructor、abortFlag、state等等
+ ncclGroupJob()：是 ncclAsyncJob 的特化/子结构，包含更多 group 相关信息

核心函数
1. ncclAsyncLaunch()：将一个任务 (job) 添加到异步队列，若当前不在 group 中则立即执行，否则延迟到 ncclGroupEnd() 统一执行
2. ncclAsyncJobMain()：异步任务的执行主函数，执行 job->func() 函数，并更新任务状态 job->state
3. ncclAsyncJobComplete()：阻塞等待该 job 对应的 pthread 线程结束 (pthread_join) 并处理结果。
4. 对外的API
   + ncclGroupStart()若上一个group的操作还未完成，则先groupJobComplete(),然后调用ncclGroupStartInternal():ncclGroupDepth++，初始化状态
   + ncclGroupEnd()：调用ncclGroupEndInternal。如果有未执行的任务（communicator / 异步任务 / 预连接任务）则：
     + 创建 ncclGroupJobMain，将相关任务信息绑定到这个 group job 上
     + 非阻塞模式 -> pthread_create 执行 groupLaunch
     + 阻塞模式 -> 直接调用 groupLaunch(&ncclGroupJobMainPtr->base)并且等待执行完毕;
5. doLaunches()：执行 group job 的实际操作，对group中每个communicator
   1. ncclLaunchPrepare,准备kernel，合并multiple ops
   2. 启动kernel
      + ncclLaunchKernelBefore_NoUncapturedCuda (在 kernel 启动前，把计算任务的参数传输到 GPU 内存)
      + ncclLaunchKernel (执行 CUDA kernel)
      + ncclLaunchKernelAfter_NoCuda (在 kernel 启动后，在 kernel 运行结束后，执行 cleanup 任务)
6. groupLaunch()：真正执行“分组”中的所有 communicator 及异步任务
   + 进行连接，给每个comm建立p2p连接，等待这些异步线程完成
   +  doLaunches（），将这些kernel plan上传到GPU，并执行


#### init.cc
1. ncclInit()进行nccl库初始化(初步的初始化，基本函数可调用，但还难以通信)
   + 初始化环境，GDRCOPY（用于直接GPU访问内存）
   + 初始化引导网络，为NCCL网络通信做准备
2. ncclGetUniqueId()：(调用bootstrapGetUniqueId())获取NCCL的唯一ID，并广播给所有rank
3. ncclCommPushCudaFree 将需要释放的CUDA资源压入队列然后由ncclDestructorFnCudaFree()释放CUDA内存 
4. commAlloc()与commFree()：管理NCCL通信器，分配和释放通信器
5. fillInfo():填充通信器的基本信息
6. setupChannel():初始化通信通道，遍历ring计算rank0的索引和其他rank的“相对索引”，设置userRank存储拓扑结构
7. computeBuffSizes():计算缓冲区大小，确定P2P通信的chunk大小
8. ncclCommInitRank/ncclCommInitAll/paseCommConfig/ncclCommDestroy
9. ncclCommCount、ncclCommCuDevice、ncclCommUserRank：通信属性查询


实际调用
1. devCommSetup():在 GPU 端分配并初始化 ncclDevCommAndChannels 结构体（用于存储 NCCL 设备端信息）
```c++
struct alignas(16) ncclDevCommAndChannels {
  struct ncclDevComm comm;
  struct ncclDevChannel channels[MAXCHANNELS];
};
```
2. collNetTrySetup():连接CollNet，计算CollNet头结点，配置CollNet发送/接受通道并且进行验证
   + ncclTransportP2pConnect()
   + ncclTransportP2pSetup() 
3. initTransportsRank()
   + 初始化 bootstrap的通信器
   + ALLGather，确保索引GPU进程可获得所有rank的信息
   + 计算同一个内的 GPU 数量 (intraRanks) 和 进程内 rank (intraRank)。
   + 构建拓扑，将GPU绑定到最接近的CPU核心减小数据传输延迟
   + 计算Ring、Tree、CollNet通道，并且设置Ring和Tree进程的P2P通道
   + 处理了 AllGather3 操作、节点和通道的设置、以及 CollNet 支持的确定
   + collNetTrySetup()
4. ncclCommInitRank()
   + 分配并且初始化NCCL通信器，调用ncclCommInitRankFunc() 

#### nccl.h.in
[nccl.h.in](nccl-2.17.1-1/src/nccl.h.in)
函数声明

#### net.cc
[net.cc](nccl-2.17.1-1/src/net.cc)
定义 v4/v5 到 v6 的适配结构，用于转换旧版本的 NCCL 网络 API，使其能适用于 v6
1. ncclNetPluginInit() ：动态加载 NCCL 网络插件，并尝试使用 v6，若不支持，则降级到 v5 或 v4 版本。
2. ncclNetInit():NCCL网络初始化，确定要使用的插件并且检查是否可用，若CollNet可用则启动CollNet
3. ncclGpuGdrSupport()：用于检测 GPU 是否支持 GPUDirect RDMA（GDR），即 GPU 直接访问网卡（NIC）的内存，
   + CUDA版本
   + 网络设备是否支持GDR
   + 建立连接，将GPU注册到网卡 

#### transport.cc
[transport.cc](nccl-2.17.1-1/src/transport.cc)
```c++
struct ncclTransport* ncclTransports[NTRANSPORTS] = {
  &p2pTransport,//P2P传输
  &shmTransport,//共享内存传输
  &netTransport,//跨节点网络传输
  &collNetTransport//集合通信传输
};
```

1. selectTransport()：选择合适的传输方式
   + 遍历 ncclTransports（4 种通信方式）
   + 调用 canConnect() 判断能否连接
   + 选择最合适的传输方式，并调用 setup() 进行配置
2. ncclTransportP2pConnect()：用于建立 点对点（P2P）连接，标记哪些 GPU 之间需要建立通信。
   + 使用 comm->connectRecv 和 comm->connectSend 记录 要接收/发送数据的目标 GPU
3. ncclTransportP2pSetup()：设置 NCCL 中的 CollNet 传输通道。**若失败则回退到P2P网络**
   + 选择transport并且初始化Connector
   + Master进程进行setup、准备建立连接、初始化连接信息
   + 交换连接信息
      + 接收端执行ALLGather进行同步信息，建立GPU通信拓扑
      + 发送端使用recv端Master发送来的connect指针
   + Master进程调用connect进行连接后将连接信息拷贝到GPU设备
   + 接收端向发送端发送连接信息，让发送端获得CollNet连接的信息
4. ncclTransportCollNetCheck()：检查 CollNet 是否成功设置，若失败则回退到 P2P 网络
5. ncclTransportCollNetFree()：释放 CollNet 资源


#### proxy.cc
[proxy.cc](nccl-2.17.1-1/src/proxy.cc) 
NCCL proxy主要用于：跨服务器的GPU通信/非P2P直连通信

代理任务pool
1. allocateArgs：为新任务分配内存，如果任务池为空，则创建新的池。链式存储任务，减少频繁的内存分配操作

代理响应
1. expectedProxyResponseStore()：储存响应。查找 是否已经存储了 opId 任务的响应。如果找到相同的任务 ID，则将 respBuff 复制到对应位置，并标记 done = true
2. expectedProxyResponseEnqueue()：等待代理返回。查找 任务 opId 是否已经完成。如果完成，将 respBuff 拷贝到返回值，并释放存储空间
3. expectedProxyResponseRemove()：删除响应。如果任务仍在进行，则抛出警告

代理出入队列
1. asyncProxyOpEnqueue()：添加任务 到 peer->asyncOps 链表中
2. asyncProxyOpDequeue()：查找任务，如果任务 opId 存在，则释放其 reqBuff 和 respBuff 并删除

调试
1. ncclDumpProxyState()：在 SIGINT 信号触发时，打印 NCCL Proxy 的任务队列状态


代理调度
1. ncclProxyOpToArgs()：将ncclProxyOp代理任务转换为ncclProxyArgs格式，便于调试、执行和调度
2. ProxyAppend()： 将一个 Proxy 任务追加到 NCCL 代理任务队列，用于调度 GPU 之间的通信。
   + 可能会尝试合并多个Proxy任务，利用ncclProxyOpToArgs(,,args->nsubs)合并 
   + nccl proxy的active任务列表式args结构体
3. ncclProxyPost（）：将Proxy任务加入到 proxyOpsPool 的任务队列中，并通知工作线程执行。
4. ncclLocalOpAppend()：将 proxyOp 任务存入 proxyOps池中，拷贝任务数据，当任务数达到一定数量时，出发ncclProxyPost发送任务
5. ncclProxySaveOp()：根据任务的模式(是Ting/Tree/CollNet等中的哪种任务模式) 这个函数 检查是否需要 Proxy 任务，并调用 SaveProxy 进行存储
6. ncclProxyComputeP2p()：计算P2P任务的参数，包括通信任务的参数，包括 步长、数据块大小、通信协议，并且进行优化
7. ncclProxyGetPostedOps()：获取并处理已经发布的 NCCL Proxy 任务。
   + 若已经有任务在执行则直接返回
   + 若nextOps有任务，则跳到process_nextops
     + 将任务从pool->nextOps中移除，并加入到active中,再将pool->nextOps设置为空
   + 否则等待新的任务提交

任务执行
1. progressOps():这个函数 遍历 NCCL Proxy 任务队列，逐个执行任务.任务完成后，调用 removeOp 清理已完成任务
2. ncclProxyProgress()：NCCL Proxy线程的主循环，负责调度和执行Proxy操作
   + 不断循环执行progressOps()执行任务，并且通过ncclProxyGetPostedOps()处理新提交的Proxy任务
3. ncclProxyStart()：遍历本地所有的 Proxy 任务，并调用 ncclProxyPost() 提交任务给 Proxy 线程，确保它们能够被执行
4. ncclProxyProgressCreate()：创建 NCCL Proxy 线程，并且设置名称用于管理 NCCL Proxy 任务的执行
5. ncclProxyNewConnect()：在pool中分配新的Proxy连接并且返回唯一的id。若满了啧新分配一个bank，从0开始计数
6. ncclProxyFreeConnections()：遍历pool的连接逐个ProxyFree，最后释放pool本身
7. ncclProxyConnect()：初始化proxyConn连接，与Proxy线程建立通信
   + ncclSocketInit()：初始化socket
   + ncclSocketSend/ncclSocketRecv()：发送/接收消息。最后接收到proxyConn->connection，完成 Proxy 连接的建立
8. ncclProxyCallBlocking()：阻塞等待Proxy任务完成，并返回结果
   + 先调用ncclProxyCallAsync: 异步调用，发送请求，返回一个opId，然后阻塞等待ncclPollProxyResponse()返回结果

任务池/线程管理
1. proxyProgressInit():创建共享内存SHM，，初始化共享pool，储存Proxy的任务管理，**启动ncclProxyProgressCreate()**
2. proxyConnInit()：初始化新的 Proxy 连接，并 在 Socket 上等待接收 连接信息。确保与 peer (其他进程) 的连接建立成功，并 通知 Proxy 任务池
3. proxyProgressAsync()：处理异步 Proxy 任务（如 setup 和 connect 操作），确保异步任务正确执行，执行完成后通知请求方
   + 任务完成，发送opId和返回的数据respBuff
   + 从异步队列中移除任务
4. proxyConnSharedInit()：初始化**共享连接**，接收并且存储channel数量，opId（用于区分任务）
5. proxyConvertFd()：处理 cuMem API 支持的文件描述符 (FD) 转换，用于 CUDA 内存跨进程共享。
6. ncclProxyService()：管理proxy连接，执行proxy任务
   + 轮询socket监听连接:poll(pollfds, NCCL_MAX_LOCAL_RANKS+1, asyncOpCount ? 0 : 500);
   + 解析proxy消息 ，可能是初始化连接，可能是建立传输连接，可能是请求停止proxy服务
7. ncclProxyCreate()/ncclProxyDestroy():创建/释放Proxy线程












































































































































































































































































































































































































































































































































































































































































































