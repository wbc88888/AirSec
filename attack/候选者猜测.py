import os
import sys
import logging
import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TABLE_PATH = os.path.join('attack', 'table', '6digit_pin.csv')
SECRET_CODE = '111000'
S = '111xx2'
T = '11211'
XS = 0
XT = 0

D_heng = np.array([
    [1, 4, 4, 3, 3, 3, 2, 2, 2, 1],
    [4, 1, 1, 2, 2, 2, 3, 3, 3, 4],
    [4, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [3, 2, 1, 1, 1, 2, 2, 2, 3, 3],
    [3, 2, 2, 1, 1, 1, 2, 2, 2, 3],
    [3, 2, 2, 2, 1, 1, 1, 2, 2, 2],
    [2, 3, 2, 2, 2, 1, 1, 1, 2, 2],
    [2, 3, 3, 2, 2, 2, 1, 1, 1, 2],
    [2, 3, 3, 3, 2, 2, 2, 1, 1, 1],
    [1, 4, 3, 3, 3, 2, 2, 2, 1, 1]
])

D_jiugong = np.array([
    [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    [2, 1, 2, 3, 2, 2, 3, 3, 3, 3],
    [2, 2, 1, 2, 2, 2, 2, 3, 3, 3],
    [2, 3, 2, 1, 3, 2, 2, 3, 3, 3],
    [3, 2, 2, 3, 1, 2, 3, 2, 2, 3],
    [3, 2, 2, 2, 2, 1, 2, 2, 2, 2],
    [3, 3, 2, 2, 3, 2, 1, 3, 2, 2],
    [4, 3, 3, 3, 2, 2, 3, 1, 2, 3],
    [4, 3, 3, 3, 2, 2, 2, 2, 1, 2],
    [4, 3, 3, 3, 3, 2, 2, 3, 2, 1]
])

def load_table(csv_path: str) -> pd.DataFrame:
    """
    加载 CSV 候选表格，必须包含列 'PIN', 'Spatial', 'Temporal'。
    """
    if not os.path.isfile(csv_path):
        logger.error(f"未找到表格文件: {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path, dtype={'PIN': str, 'Spatial': str, 'Temporal': str})
    except Exception as e:
        logger.error(f"加载 CSV 失败: {e}")
        sys.exit(1)

    required_cols = {'PIN', 'Spatial', 'Temporal'}
    if not required_cols.issubset(df.columns):
        logger.error(f"CSV 缺少必要列: {required_cols - set(df.columns)}")
        sys.exit(1)
    return df


def make_pro_tprop(prop, idx):
    L = len(prop)
    new_prop = prop[0:idx] + '-' + prop[idx+1:L]

    known_indices = []
    # 记录已知数字位置
    for idx, val in enumerate(new_prop):
        if val != 'x' and val != 'y' and val != 'z' and val != '-':
            known_indices.append(idx)

    nums = [int(new_prop[idx]) for idx in known_indices]
    unique_sorted = sorted(set(nums))
    rank = {v: str(i + 1) for i, v in enumerate(unique_sorted)}
    
    real_new_prop = ''
    for i in range(L):
        if i in known_indices:
            real_new_prop = real_new_prop + rank[int(new_prop[i])]
        else:
            real_new_prop = real_new_prop + new_prop[i]
    return real_new_prop


def make_pro_sprop(prop, idx):
    L = len(prop)
    new_prop = prop[0:idx] + '-' + prop[idx+1:L]

    known_indices = []
    # 记录已知数字位置
    for idx, val in enumerate(new_prop):
        if val != 'x' and val != 'y' and val != 'z' and val != '-':
            known_indices.append(idx)

    nums = [int(new_prop[idx]) for idx in known_indices]
    unique_sorted = list(dict.fromkeys(nums))
    rank = {v: str(i + 1) for i, v in enumerate(unique_sorted)}
    
    real_new_prop = ''
    for i in range(L):
        if i in known_indices:
            real_new_prop = real_new_prop + rank[int(new_prop[i])]
        else:
            real_new_prop = real_new_prop + new_prop[i]
    return real_new_prop


def make_pro_sprop_list(sprop, xs):
    pro_sprop_list = [sprop]
    where_max_unknow_spropidx_list = [-1]

    for i in range(xs):
        max_j = len(pro_sprop_list)
        for j in range(max_j):
            for p in range(where_max_unknow_spropidx_list[0]+1, 6):
                pro_sprop = make_pro_sprop(pro_sprop_list[0], p)
                pro_sprop_list.append(pro_sprop)
                where_max_unknow_spropidx_list.append(p)
            pro_sprop_list.pop(0)
            where_max_unknow_spropidx_list.pop(0)

    return pro_sprop_list


def make_pro_tprop_list(tprop, xt):
    pro_tprop_list = [tprop]
    where_max_unknow_tpropidx_list = [-1]

    for i in range(xt):
        max_j = len(pro_tprop_list)
        for j in range(max_j):
            for p in range(where_max_unknow_tpropidx_list[0]+1, 5):
                pro_tprop = make_pro_tprop(pro_tprop_list[0], p)
                pro_tprop_list.append(pro_tprop)
                where_max_unknow_tpropidx_list.append(p)
            pro_tprop_list.pop(0)
            where_max_unknow_tpropidx_list.pop(0)

    return pro_tprop_list


def make_pro_prop_list(sprop, tprop, xs, xt):
    pro_sprop_list, pro_tprop_list = [sprop], [tprop]
    where_max_unknow_spropidx_list, where_max_unknow_tpropidx_list = [-1], [-1]

    for i in range(xs):
        max_j = len(pro_sprop_list)
        for j in range(max_j):
            for p in range(where_max_unknow_spropidx_list[0]+1, 6):
                pro_sprop = make_pro_sprop(pro_sprop_list[0], p)
                pro_sprop_list.append(pro_sprop)
                where_max_unknow_spropidx_list.append(p)
            pro_sprop_list.pop(0)
            where_max_unknow_spropidx_list.pop(0)

    for i in range(xt):
        max_j = len(pro_tprop_list)
        for j in range(max_j):
            for p in range(where_max_unknow_tpropidx_list[0]+1, 5):
                pro_tprop = make_pro_tprop(pro_tprop_list[0], p)
                pro_tprop_list.append(pro_tprop)
                where_max_unknow_tpropidx_list.append(p)
            pro_tprop_list.pop(0)
            where_max_unknow_tpropidx_list.pop(0)

    return pro_sprop_list, pro_tprop_list

    
def filter_candidates_tprop(pro_tprop_list, table_df):
    candidates = []
    for idx, text in enumerate(table_df['Temporal']):
        pro_text_list = make_pro_tprop_list(text, XT)
        if bool(set(pro_text_list) & set(pro_tprop_list)):
            candidates.append(idx)
    
    return candidates


def filter_candidates_sprop(pro_sprop_list, cand_tprop, table_df):
    candidates = []
    for idx in cand_tprop:
        text = table_df['Spatial'][idx]

        for pro_sprop in pro_sprop_list:
            glue_ = False
            unknown_indices = []
            glue_indices = []
            # 先检查sprop状况
            for i, val in enumerate(pro_sprop):
                if val == 'x' or val == 'y' or val == 'z':
                    glue_ = True
                    unknown_indices.append(i)
                    glue_indices.append(i)
                elif val == '-':
                    unknown_indices.append(i)

            # 无xyz情况下同时间特征处理
            if glue_ == False:
                pro_text_list = make_pro_sprop_list(text, XS)
                if pro_sprop in pro_text_list:
                    candidates.append(idx)
                    break
            # 有xyz情况下
            else:
                # 仅粘合位置需要数字相连约束
                ok = True
                idx_str = f'{idx:06d}'
                for i in glue_indices:
                    for j in glue_indices:
                        if j <= i or pro_sprop[i] != pro_sprop[j]:
                            continue
                        d1 = int(idx_str[i])
                        d2 = int(idx_str[j])

                        if D_heng[d1][d2] > 1 or D_heng[d2][d1] > 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

                # 拼接已知位置字符
                known_chars = ''.join(c for i, c in enumerate(idx_str) if i not in unknown_indices)
                known_sprop = ''.join(c for i, c in enumerate(pro_sprop) if i not in unknown_indices)

                # 生成实际空间编码
                labels = []
                mapping = {}
                next_label = 1
                for ch in known_chars:
                    if ch not in mapping:
                        mapping[ch] = str(next_label)
                        next_label += 1
                    labels.append(mapping[ch])
                generated_sprop = ''.join(labels)

                if sum(x != y for x, y in zip(generated_sprop, known_sprop)) <= XS:
                    candidates.append(idx)
                    break

    return [f'{i:06d}' for i in candidates]


def main():
    table_df = load_table(TABLE_PATH)  # 加载表格

    pro_sprop_list, pro_tprop_list = make_pro_prop_list(S, T, XS, XT)  # 返回可能时空特征值列表

    cand_tprop = filter_candidates_tprop(pro_tprop_list, table_df)  # 先由时间特征筛选一批

    candidates = filter_candidates_sprop(pro_sprop_list, cand_tprop, table_df)  # 再有空间特征筛选一批

    print(candidates)

    for rank, code in enumerate(candidates, start=1):
        if code == SECRET_CODE:
            logger.info(f"在第{rank}次猜中，候选者列表大小: {len(candidates)}")
            sys.exit(0)

    logger.info("没有猜中")

if __name__ == '__main__':
    main()
