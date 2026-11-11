/*************************************************************************
 * P2P 细粒度传输日志模块 - 实现
 *
 * ADD
 * Author: zhangzhao
 * Date:
 * 针对 NCCL 各集合通信算法（Ring AllReduce / AllGather / ReduceScatter /
 * Broadcast / Reduce 以及 Tree AllReduce），在 CPU 侧模拟其逐步执行过程，
 * 把每一次底层 GPU-to-GPU send 操作单独记录到 mockOutput->p2pTraces。
 *
 * 设计要点：
 *   1. 每一次 send 都单独记录一行，不合并、不省略；
 *   2. 多 channel 并行时各 channel 的 send 独立枚举；
 *   3. 数据量按 chunk 精确计算，尾部不满 chunk 时按余量计算；
 *   4. Ring 拓扑读 mockOutput->ringChannels，Tree 拓扑读 mockOutput->treeChannels；
 *   5. CollNet/NVLS 等使用专用硬件通道的算法暂不纳入 P2P trace。
 *
 * 移植说明：为兼容 NCCL 2.29.7 的 ncclTaskColl 架构，主入口与各算法函数
 *   改为接收显式标量参数（coll/algorithm/chunkCount/datatype/root），
 *   不再依赖 2.20.5 的 ncclInfo 结构。
 ************************************************************************/
#include "p2p_trace.h"
#include "debug.h"
#include "collectives.h"  // ncclTypeSize
#include "device.h"       // NCCL_ALGO_* / ncclFunc_t
#include <algorithm>
#include <vector>

/* 获取指定 channel、rank 的 ring 下一跳 */
static int getRingNext(struct ncclMockOutput *m, int c, int r) {
    return m->ringChannels[c][r][1]; // index 1 = next
}
/* 获取指定 channel、rank 的 ring 前一跳 */
static int getRingPrev(struct ncclMockOutput *m, int c, int r) {
    return m->ringChannels[c][r][0]; // index 0 = prev
}

/* 向 p2pTraces 追加一条记录 */
static void addP2pTrace(struct ncclMockOutput *m, int coll, int algo,
                        int channel, int step, int src, int dst,
                        size_t bytes, int chunkId) {
    struct ncclMockOutput::P2pTransfer t;
    t.collective = coll;
    t.algorithm  = algo;
    t.channel    = channel;
    t.step       = step;
    t.srcRank    = src;
    t.dstRank    = dst;
    t.dataBytes  = bytes;
    t.chunkId    = chunkId;
    m->p2pTraces.push_back(t);
}

/* ========================================================================
 * Ring AllReduce：ReduceScatter（nranks-1步）+ AllGather（nranks-1步）
 * 每步所有 rank 同时向 ring.next 发送一个 chunk。
 * ======================================================================== */
static ncclResult_t traceRingAllReduce(struct ncclComm *comm, int chunkCount,
                                       ncclDataType_t datatype, size_t workCount, int nChannels) {
    struct ncclMockOutput *m = comm->mockOutput;
    int nRanks = comm->nRanks;
    uint32_t typeSize = ncclTypeSize(datatype);
    int globalStep = 0;
    size_t loopCount = (size_t)nRanks * chunkCount;

    for (int c = 0; c < nChannels; c++) {
        if (m->ringChannels.find(c) == m->ringChannels.end()) continue;
        size_t channelCount = workCount;
        for (size_t off = 0; off < channelCount; off += loopCount) {
            size_t rem = channelCount - off;
            int curCC = chunkCount;
            if (rem < loopCount) curCC = (int)((rem + nRanks - 1) / nRanks);
            size_t bytes = (size_t)std::min((size_t)curCC, rem) * typeSize;
            int baseId = (int)(off / loopCount) * 2 * (nRanks - 1);

            /* Phase 1: ReduceScatter，nranks-1 步 */
            for (int s = 0; s < nRanks - 1; s++) {
                for (int r = 0; r < nRanks; r++) {
                    addP2pTrace(m, ncclFuncAllReduce, NCCL_ALGO_RING,
                                c, globalStep, r, getRingNext(m, c, r),
                                bytes, baseId + s);
                }
                globalStep++;
            }
            /* Phase 2: AllGather，nranks-1 步 */
            for (int s = 0; s < nRanks - 1; s++) {
                for (int r = 0; r < nRanks; r++) {
                    addP2pTrace(m, ncclFuncAllReduce, NCCL_ALGO_RING,
                                c, globalStep, r, getRingNext(m, c, r),
                                bytes, baseId + (nRanks - 1) + s);
                }
                globalStep++;
            }
        }
    }
    INFO(NCCL_MOCK_ROOT, "P2P Trace: Ring AllReduce nRanks=%d nChannels=%d total=%zu",
         nRanks, nChannels, m->p2pTraces.size());
    return ncclSuccess;
}

/* ========================================================================
 * Ring AllGather：nranks-1 步，每步所有 rank 同时向 ring.next 发送一个 chunk
 * ======================================================================== */
static ncclResult_t traceRingAllGather(struct ncclComm *comm, int chunkCount,
                                       ncclDataType_t datatype, size_t workCount, int nChannels) {
    struct ncclMockOutput *m = comm->mockOutput;
    int nRanks = comm->nRanks;
    uint32_t typeSize = ncclTypeSize(datatype);
    int globalStep = 0;

    for (int c = 0; c < nChannels; c++) {
        if (m->ringChannels.find(c) == m->ringChannels.end()) continue;
        size_t channelCount = workCount;
        for (size_t off = 0; off < channelCount; off += (size_t)chunkCount) {
            int nelem = (int)std::min((size_t)chunkCount, channelCount - off);
            size_t bytes = (size_t)nelem * typeSize;
            int chunkId = (int)(off / chunkCount);
            for (int s = 0; s < nRanks - 1; s++) {
                for (int r = 0; r < nRanks; r++) {
                    addP2pTrace(m, ncclFuncAllGather, NCCL_ALGO_RING,
                                c, globalStep, r, getRingNext(m, c, r),
                                bytes, chunkId * (nRanks - 1) + s);
                }
                globalStep++;
            }
        }
    }
    INFO(NCCL_MOCK_ROOT, "P2P Trace: Ring AllGather nRanks=%d nChannels=%d total=%zu",
         nRanks, nChannels, m->p2pTraces.size());
    return ncclSuccess;
}

/* ========================================================================
 * Ring ReduceScatter：nranks-1 步，每步所有 rank 同时向 ring.next 发送一个 chunk
 * ======================================================================== */
static ncclResult_t traceRingReduceScatter(struct ncclComm *comm, int chunkCount,
                                           ncclDataType_t datatype, size_t workCount, int nChannels) {
    struct ncclMockOutput *m = comm->mockOutput;
    int nRanks = comm->nRanks;
    uint32_t typeSize = ncclTypeSize(datatype);
    int globalStep = 0;

    for (int c = 0; c < nChannels; c++) {
        if (m->ringChannels.find(c) == m->ringChannels.end()) continue;
        size_t channelCount = workCount;
        for (size_t off = 0; off < channelCount; off += (size_t)chunkCount) {
            int nelem = (int)std::min((size_t)chunkCount, channelCount - off);
            size_t bytes = (size_t)nelem * typeSize;
            int chunkId = (int)(off / chunkCount);
            for (int s = 0; s < nRanks - 1; s++) {
                for (int r = 0; r < nRanks; r++) {
                    addP2pTrace(m, ncclFuncReduceScatter, NCCL_ALGO_RING,
                                c, globalStep, r, getRingNext(m, c, r),
                                bytes, chunkId * (nRanks - 1) + s);
                }
                globalStep++;
            }
        }
    }
    INFO(NCCL_MOCK_ROOT, "P2P Trace: Ring ReduceScatter nRanks=%d nChannels=%d total=%zu",
         nRanks, nChannels, m->p2pTraces.size());
    return ncclSuccess;
}

/* ========================================================================
 * Ring Broadcast：数据从 root 沿 ring 逐跳传递，nranks-1 步，每步只有1次发送
 * ======================================================================== */
static ncclResult_t traceRingBroadcast(struct ncclComm *comm, int chunkCount,
                                       ncclDataType_t datatype, int root, size_t workCount, int nChannels) {
    struct ncclMockOutput *m = comm->mockOutput;
    int nRanks = comm->nRanks;
    uint32_t typeSize = ncclTypeSize(datatype);
    int globalStep = 0;

    for (int c = 0; c < nChannels; c++) {
        if (m->ringChannels.find(c) == m->ringChannels.end()) continue;
        size_t channelCount = workCount;
        for (size_t off = 0; off < channelCount; off += (size_t)chunkCount) {
            int nelem = (int)std::min((size_t)chunkCount, channelCount - off);
            size_t bytes = (size_t)nelem * typeSize;
            int chunkId = (int)(off / chunkCount);
            int sender = root;
            for (int s = 0; s < nRanks - 1; s++) {
                int next = getRingNext(m, c, sender);
                addP2pTrace(m, ncclFuncBroadcast, NCCL_ALGO_RING,
                            c, globalStep, sender, next, bytes, chunkId);
                sender = next;
                globalStep++;
            }
        }
    }
    INFO(NCCL_MOCK_ROOT, "P2P Trace: Ring Broadcast root=%d nChannels=%d total=%zu",
         root, nChannels, m->p2pTraces.size());
    return ncclSuccess;
}

/* ========================================================================
 * Ring Reduce：数据沿 ring 汇聚到 root，nranks-1 步，每步只有1次发送
 *   起始发送者：距 root 最远（沿 ring 方向 root 的前 nranks-1 跳）
 * ======================================================================== */
static ncclResult_t traceRingReduce(struct ncclComm *comm, int chunkCount,
                                    ncclDataType_t datatype, int root, size_t workCount, int nChannels) {
    struct ncclMockOutput *m = comm->mockOutput;
    int nRanks = comm->nRanks;
    uint32_t typeSize = ncclTypeSize(datatype);
    int globalStep = 0;

    for (int c = 0; c < nChannels; c++) {
        if (m->ringChannels.find(c) == m->ringChannels.end()) continue;
        size_t channelCount = workCount;
        for (size_t off = 0; off < channelCount; off += (size_t)chunkCount) {
            int nelem = (int)std::min((size_t)chunkCount, channelCount - off);
            size_t bytes = (size_t)nelem * typeSize;
            int chunkId = (int)(off / chunkCount);
            int sender = getRingPrev(m, c, root);
            for (int i = 0; i < nRanks - 2; i++) sender = getRingPrev(m, c, sender);
            int cur = sender;
            for (int s = 0; s < nRanks - 1; s++) {
                int next = getRingNext(m, c, cur);
                addP2pTrace(m, ncclFuncReduce, NCCL_ALGO_RING,
                            c, globalStep, cur, next, bytes, chunkId);
                cur = next;
                globalStep++;
            }
        }
    }
    INFO(NCCL_MOCK_ROOT, "P2P Trace: Ring Reduce root=%d nChannels=%d total=%zu",
         root, nChannels, m->p2pTraces.size());
    return ncclSuccess;
}

/* ========================================================================
 * Tree AllReduce：Reduce（上行：叶到根）+ Broadcast（下行：根到叶）
 *   treeChannels[c][r] = {up, down[0], down[1], down[2]}, up/down=-1 表示无该节点
 *   BFS策略：每轮扫描所有rank，子节点都就绪的才发送
 * ======================================================================== */
static ncclResult_t traceTreeAllReduce(struct ncclComm *comm, int chunkCount,
                                       ncclDataType_t datatype, size_t workCount, int nChannels) {
    struct ncclMockOutput *m = comm->mockOutput;
    int nRanks = comm->nRanks;
    uint32_t typeSize = ncclTypeSize(datatype);
    int globalStep = 0;

    for (int c = 0; c < nChannels; c++) {
        if (m->treeChannels.find(c) == m->treeChannels.end()) continue;
        size_t channelCount = workCount;
        for (size_t off = 0; off < channelCount; off += (size_t)chunkCount) {
            int nelem = (int)std::min((size_t)chunkCount, channelCount - off);
            size_t bytes = (size_t)nelem * typeSize;
            int chunkId = (int)(off / chunkCount);

            /* Phase 1: Reduce（叶到根） */
            std::vector<bool> sent(nRanks, false);
            int sentCnt = 0;
            int rootRank = -1;
            for (int r = 0; r < nRanks; r++) {
                if (m->treeChannels[c][r][0] == -1) {
                    sent[r] = true; sentCnt++; rootRank = r;
                }
            }
            while (sentCnt < nRanks) {
                for (int r = 0; r < nRanks; r++) {
                    if (sent[r]) continue;
                    auto &t = m->treeChannels[c][r];
                    int up = t[0], d0 = t[1], d1 = t[2], d2 = t[3];
                    bool ready = true;
                    if (d0 != -1 && !sent[d0]) ready = false;
                    if (d1 != -1 && !sent[d1]) ready = false;
                    if (d2 != -1 && !sent[d2]) ready = false;
                    if (ready && up != -1) {
                        addP2pTrace(m, ncclFuncAllReduce, NCCL_ALGO_TREE,
                                    c, globalStep, r, up, bytes, chunkId);
                        sent[r] = true; sentCnt++;
                    }
                }
                globalStep++;
            }

            /* Phase 2: Broadcast（根到叶） */
            std::vector<bool> recv(nRanks, false);
            recv[rootRank] = true;
            int recvCnt = 1;
            while (recvCnt < nRanks) {
                for (int r = 0; r < nRanks; r++) {
                    if (!recv[r]) continue;
                    auto &t = m->treeChannels[c][r];
                    int d0 = t[1], d1 = t[2], d2 = t[3];
                    if (d0 != -1 && !recv[d0]) {
                        addP2pTrace(m, ncclFuncAllReduce, NCCL_ALGO_TREE,
                                    c, globalStep, r, d0, bytes, chunkId);
                        recv[d0] = true; recvCnt++;
                    }
                    if (d1 != -1 && !recv[d1]) {
                        addP2pTrace(m, ncclFuncAllReduce, NCCL_ALGO_TREE,
                                    c, globalStep, r, d1, bytes, chunkId);
                        recv[d1] = true; recvCnt++;
                    }
                    if (d2 != -1 && !recv[d2]) {
                        addP2pTrace(m, ncclFuncAllReduce, NCCL_ALGO_TREE,
                                    c, globalStep, r, d2, bytes, chunkId);
                        recv[d2] = true; recvCnt++;
                    }
                }
                globalStep++;
            }
        }
    }
    INFO(NCCL_MOCK_ROOT, "P2P Trace: Tree AllReduce nRanks=%d nChannels=%d total=%zu",
         nRanks, nChannels, m->p2pTraces.size());
    return ncclSuccess;
}

/* ========================================================================
 * 主入口：按 coll + algorithm 分发
 * ======================================================================== */
ncclResult_t traceCollP2pOps(struct ncclComm *comm, int coll, int algorithm,
                             int chunkCount, ncclDataType_t datatype, int root,
                             size_t workCount, int nChannels) {
    if (comm->mockOutput == nullptr) return ncclSuccess;
    INFO(NCCL_MOCK_ROOT,
         "P2P Trace start: coll=%d algo=%d nChannels=%d workCount=%zu chunkCount=%d",
         coll, algorithm, nChannels, workCount, chunkCount);

    switch (coll) {
    case ncclFuncAllReduce:
        if (algorithm == NCCL_ALGO_RING)
            return traceRingAllReduce(comm, chunkCount, datatype, workCount, nChannels);
        if (algorithm == NCCL_ALGO_TREE)
            return traceTreeAllReduce(comm, chunkCount, datatype, workCount, nChannels);
        INFO(NCCL_MOCK_ROOT, "P2P Trace: AllReduce algo=%d not traced", algorithm);
        break;
    case ncclFuncAllGather:
        if (algorithm == NCCL_ALGO_RING)
            return traceRingAllGather(comm, chunkCount, datatype, workCount, nChannels);
        INFO(NCCL_MOCK_ROOT, "P2P Trace: AllGather algo=%d not traced", algorithm);
        break;
    case ncclFuncReduceScatter:
        if (algorithm == NCCL_ALGO_RING)
            return traceRingReduceScatter(comm, chunkCount, datatype, workCount, nChannels);
        INFO(NCCL_MOCK_ROOT, "P2P Trace: ReduceScatter algo=%d not traced", algorithm);
        break;
    case ncclFuncBroadcast:
        if (algorithm == NCCL_ALGO_RING)
            return traceRingBroadcast(comm, chunkCount, datatype, root, workCount, nChannels);
        INFO(NCCL_MOCK_ROOT, "P2P Trace: Broadcast algo=%d not traced", algorithm);
        break;
    case ncclFuncReduce:
        if (algorithm == NCCL_ALGO_RING)
            return traceRingReduce(comm, chunkCount, datatype, root, workCount, nChannels);
        INFO(NCCL_MOCK_ROOT, "P2P Trace: Reduce algo=%d not traced", algorithm);
        break;
    default:
        INFO(NCCL_MOCK_ROOT, "P2P Trace: collective %d not traced", coll);
        break;
    }
    return ncclSuccess;
}
