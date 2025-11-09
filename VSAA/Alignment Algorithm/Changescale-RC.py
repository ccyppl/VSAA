# 开发时间：2025/8/22 15:52
import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict, Counter
import time
from PatternMatching import match_index
from DC import rc_calibrate_point



def read_column_from_excel(filepath, column_name):
    df = pd.read_excel(filepath)
    if isinstance(column_name, int):
        return df.iloc[:, column_name].dropna().tolist()
    else:
        return df[column_name].dropna().tolist()


def save_matches_to_excel(image_seq, accel_seq, matches, output_path):
    data = []
    for img_idx, acc_idx in matches:
        data.append({
            '图像点索引': img_idx,
            '图像位置': image_seq[img_idx],
            '加速度点索引': acc_idx,
            '加速度位置': accel_seq[acc_idx]
        })
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"匹配结果已保存到：{output_path}")


from collections import defaultdict
from itertools import product

def bidirectional_segment_alignment(
    image_seq,
    accel_seq,
    threshold,
    k=3,
    start_pair=(0, 0),
    end_pair=None,
):
    """
    双端驱动的分段(2k-1)对齐：

    返回:
        matches: List[ (image_idx, accel_idx) ] 按图像索引升序
    """
    assert end_pair is not None, "必须提供 end_pair（终点已对齐）"

    n_img = len(image_seq)
    n_acc = len(accel_seq)

    # ====== 已匹配表（锚定端点） ======
    matched = dict()
    matched[start_pair[0]] = start_pair[1]
    matched[end_pair[0]] = end_pair[1]

    # —— 工具函数 —— #
    def find_candidates(img_idx):
        """返回满足阈值的加速度点索引候选（按索引升序）。"""
        x = image_seq[img_idx]
        return [j for j, a in enumerate(accel_seq) if abs(a - x) <= threshold]

    def monotonic_combos(candidate_lists, increasing=True):

        for combo in product(*candidate_lists):
            if increasing:
                if list(combo) == sorted(combo):
                    yield combo
            else:
                if list(combo) == sorted(combo, reverse=True):
                    yield combo

    def resolve_votes(votes):

        mode_candidates = [[(v, 1.0) for v in set(votes)]]
        rc_res = rc_calibrate_point(mode_candidates, K1=None)
        return rc_res["s_star"]

    def best_combo_for_pattern(patt_img_idxs, anchor_fixed=None, increasing=True):

        candidate_sets = []
        for t, img_idx in enumerate(patt_img_idxs):
            if anchor_fixed is not None and t == anchor_fixed[0]:
                candidate_sets.append([anchor_fixed[1]])
                continue
            cand = find_candidates(img_idx)
            if not cand:
                return None  # 该模式失败
            candidate_sets.append(cand)

        # 穷举单调组合，取 PMI 最大
        best_score, best_combo = -1e18, None
        for combo in monotonic_combos(candidate_sets, increasing=increasing):
            acc_pts = [accel_seq[j] for j in combo]
            img_pts = [image_seq[i] for i in patt_img_idxs]
            score = match_index(img_pts, acc_pts)
            if score > best_score:
                best_score, best_combo = score, combo
        return best_combo

    def run_block(block_img_indices, anchor_side="left"):

        L = list(block_img_indices)
        if anchor_side == "right":
            L = list(reversed(L))

        if L[0] not in matched:
            return {}

        votes_map = defaultdict(list)

        # --- 定义三类模式 ---
        front_k = L[:k]
        back_k  = L[k-1:]

        step = (len(L) - 1) // (k - 1)
        cross  = [L[i] for i in range(0, len(L), step)][:k]

        # === 1) 前 k：以段首为锚点 ===
        acc_anchor_idx = matched[L[0]]
        combo = best_combo_for_pattern(
            front_k,
            anchor_fixed=(0, acc_anchor_idx),
            increasing=(anchor_side != "right")
        )
        if combo is not None:
            for img_idx, acc_idx in zip(front_k, combo):
                votes_map[img_idx].append(acc_idx)

            matched[front_k[-1]] = combo[-1]

        if back_k[0] in matched:
            combo = best_combo_for_pattern(
                back_k,
                anchor_fixed=(0, matched[back_k[0]]),
                increasing=(anchor_side != "right")
            )
            if combo is not None:
                for img_idx, acc_idx in zip(back_k, combo):
                    votes_map[img_idx].append(acc_idx)

        combo = best_combo_for_pattern(
            cross,
            anchor_fixed=None,
            increasing=(anchor_side != "right")
        )
        if combo is not None:
            for img_idx, acc_idx in zip(cross, combo):
                votes_map[img_idx].append(acc_idx)

        cross_targets = set(cross[1:])
        for img_idx in cross_targets:
            if img_idx in votes_map and len(votes_map[img_idx]) >= 2:
                chosen = resolve_votes(votes_map[img_idx])

                votes_map[img_idx] = [chosen]
                matched[img_idx] = chosen


        for img_idx, vlist in votes_map.items():
            if img_idx not in matched and len(vlist) >= 1:
                matched[img_idx] = vlist[0]

        return votes_map

    # ===== 分两端推进（段长 2k-1） =====
    start_img, end_img = start_pair[0], end_pair[0]
    mid = (start_img + end_img) // 2

    # ---- 从起点向中间推进（左 -> 右）----
    seg_start = start_img
    while seg_start <= mid:
        seg_end = seg_start + (2*k - 2)  # inclusive，长度=2k-1
        if seg_end > end_img:
            seg_start = max(seg_start, end_img - (2*k - 2))
            seg_end = end_img

        block = list(range(seg_start, seg_end + 1))
        _ = run_block(block, anchor_side="left")

        # 下一段：滑动 2k-2 个点
        seg_start += (2*k - 2)
        if seg_start >= end_img:
            break
        # 段首应已匹配，否则无法继续推进
        if seg_start not in matched:
            break

    # ---- 从终点向中间推进（右 -> 左）----
    seg_end = end_img
    while seg_end >= mid:
        seg_start = seg_end - (2*k - 2)
        if seg_start < start_img:
            seg_start = start_img
            seg_end = min(seg_start + (2*k - 2), end_img)

        block = list(range(seg_start, seg_end + 1))
        _ = run_block(block, anchor_side="right")

        seg_end -= (2*k - 2)
        if seg_end <= start_img:
            break
        if seg_end not in matched:
            break

    # 输出配对（按图像索引排序）
    return sorted(matched.items(), key=lambda x: x[0])



# ========== 示例主程序 ==========

if __name__ == "__main__":
    # 路径请按需修改
    image_path = '冲击指数峰值点/图像特征点.xlsx'
    accel_path = '冲击指数峰值点/待对齐加速度数据.xlsx'
    output_path = 'result/k=4/变尺度对齐.xlsx'

    # 读取数据
    image_seq = read_column_from_excel(image_path, '图像特征点')
    accel_seq = read_column_from_excel(accel_path, '精确位置')

    # 参数
    threshold = 5
    k = 4
    start_pair = (0, 0)
    end_pair = (len(image_seq) - 1, len(accel_seq) - 1)


    t0 = time.perf_counter()
    matches = bidirectional_segment_alignment(
        image_seq=image_seq,
        accel_seq=accel_seq,
        threshold=threshold,
        k=k,
        start_pair=start_pair,
        end_pair=end_pair,

    )
    t1 = time.perf_counter()
    print(f"运行时间：{t1 - t0:.6f} 秒")
    print(f"已匹配点数：{len(matches)} / {len(image_seq)}")

    save_matches_to_excel(image_seq, accel_seq, matches, output_path)