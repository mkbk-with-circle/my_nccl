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


## ncclGetUniqueId
1. 调用ncclInit()进行nccl库初始化(初步的初始化，基本函数可调用，但还难以通信)
   + 初始化环境，GPU
   + 初始化引导网络，为NCCL网络通信做准备
   + ![init.cc](nccl-master/src/init.cc)
2. 调用bootstrapGetUniqueId()函数来获取一个唯一的ID。
   + 包括两部分：一个随机数+一个环境变量（若无则是bootstrap的网络地址） 
   + ![bootstrap.cc](nccl-master/src/bootstrap.cc)


## 初始化通信器ncclCommInitRank
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


## bootstrapInit()
![bootstrap.cc](nccl-master/src/bootstrap.cc)
利用已知的rank0网络地址（UniqueId），建立环形网络，allgather获取所有rank的信息
1. 函数输入ncclUniqueId，从而获得ncclUniqueId中包含的rank0的网络地址，每个rank上都有rank0的网络地址
2. 所有rank根据rank0的网络地址，建立socket并向rank0发送自己的网络地址，rank0上现在就有所有rank的网络地址了
3. rank0告诉每个rank它的下一个节点网络地址，完成**环形网络**建立（方便进行通信）
4. AllGather全局收集所有节点的网络地址













