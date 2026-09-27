---
name: cpu-hpc-skill
description: bitmatrix vcnt.
---

只改 kernel.cpp。禁止线程和 omp。第一轮用写文件工具写入，回复里的代码不会落盘。只跑 `bash tools/test_candidate.sh`。更快的正确版本留下。

一行 N/8 字节。块数是 N/128，不是 N/16。N%128!=0 时最后 8 字节用 16 项半字节表。公开用例看不出漏尾。

小端：列 j 是字 j/64 的第 j%64 位，字节最低位是更小列号。mask 相同。result 先全 0。用 __builtin_ctzll 收集选中行，按该数组每 8 个一组，不足补 0。不要按物理行分组。

每块 8 行 vld1q_u8，字节内转置后 vcntq_u8。寄存器 c 的 lane L 是列 (byte_offset+L)*8+c。交换 t=((x>>s)^y)&m; x^=t<<s; y^=t。s=1,m=0x55 对 (0,1)(2,3)(4,5)(6,7)。s=2,m=0x33 对 (0,2)(1,3)(4,6)(5,7)。s=4,m=0x0F 对 (0,4)(1,5)(2,6)(3,7)。

uint8 累加，每 30 组 vst1q_u8 到临时数组再写入 result 并清零。禁止 vgetq_lane_u8。编译选项留空。
