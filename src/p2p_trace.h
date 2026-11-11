/*************************************************************************
 * P2P 细粒度传输日志模块 - 头文件
 *
 * ADD
 * Author: zhangzhao
 * Date:
 * 功能：基于已计算完成的通信计划参数（算法、拓扑、chunk分块），
 *       在CPU侧逐步模拟集合通信算法的执行过程，枚举并记录
 *       每一次底层GPU间P2P send操作到 mockOutput->p2pTraces。
 *
 * 用途：日志导出后供 ns-3 仿真平台解析，复现通信流程。
 *       每条记录对应一次独立的 GPU-to-GPU 数据传输。
 *
 * 说明：移植到 NCCL 2.29.7 时，为解耦 2.20.5 的 ncclInfo / 2.29.7 的
 *       ncclTaskColl 结构差异，主入口改为接收显式标量参数。
 ************************************************************************/
#ifndef NCCL_P2P_TRACE_H_
#define NCCL_P2P_TRACE_H_

#include "comm.h"
#include "nccl.h"

/*
 * traceCollP2pOps - P2P日志生成主入口
 * 在集合通信计划生成完成后调用，根据 coll 类型和 algorithm
 * 分发到对应的算法模拟函数。
 *
 * @param comm       通信器（提供 mockOutput 中的拓扑和 nRanks）
 * @param coll       集合通信类型（ncclFunc_t）
 * @param algorithm  算法（NCCL_ALGO_RING / NCCL_ALGO_TREE ...）
 * @param chunkCount 单个 chunk 的元素数
 * @param datatype   数据类型（用于计算字节数）
 * @param root       root rank（Broadcast/Reduce 使用）
 * @param workCount  每个channel分配的数据元素数
 * @param nChannels  实际使用的通道数
 */
ncclResult_t traceCollP2pOps(struct ncclComm *comm, int coll, int algorithm,
                             int chunkCount, ncclDataType_t datatype, int root,
                             size_t workCount, int nChannels);

#endif // NCCL_P2P_TRACE_H_
