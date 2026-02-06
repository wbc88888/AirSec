import pandas as pd
import numpy as np

# 计算距离矩阵 D
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

# 计算 PIN 候选的时间属性
def get_t(target, D):
    s = [D[int(target[i])][int(target[i + 1])] for i in range(len(target) - 1)]
    usets = set(s)

    # 攻击者猜测实际的键间距，简化为 1, 2, 3, 4 的映射
    if 1 not in usets and 2 in usets and 3 in usets and 4 in usets:
        s = [1 if x == 2 else 2 if x == 3 else 3 if x == 4 else x for x in s]
    elif 2 not in usets and 1 in usets and 3 in usets and 4 in usets:
        s = [2 if x == 3 else 3 if x == 4 else x for x in s]
    elif 3 not in usets and 1 in usets and 2 in usets and 4 in usets:
        s = [3 if x == 4 else x for x in s]
    elif min(usets) == 2 and 1 not in usets and 3 not in usets:
        s = [1 if x == 2 else 2 for x in s]
    elif min(usets) == 3 and 1 not in usets and 2 not in usets:
        s = [1 if x == 3 else 2 for x in s]
    elif min(usets) == 1 and 2 not in usets and 4 not in usets:
        s = [2 if x == 3 else x for x in s]
    elif min(usets) == 1 and 2 not in usets and 3 not in usets:
        s = [2 if x == 4 else x for x in s]
    elif min(usets) == 2 and 1 not in usets and 4 not in usets:
        s = [1 if x == 2 else 2 if x == 3 else x for x in s]
    elif min(usets) == 3 and 1 not in usets and 2 not in usets:
        s = [1 if x == 3 else 2 for x in s]
    elif min(usets) == 4 and all(x not in usets for x in [1, 2, 3]):
        s = [1 if x == 4 else x for x in s]

    return ''.join(map(str, s))

# 预先分配列表来存储 PIN 和 Spatial 数据
pins = []
spatials = []
temporal = []

for i in range(1000000):
    # 使用格式化字符串生成 6 位数的 PIN
    num = f"{i:06d}"
    
    # 计算空间属性 Spatial
    Spatial = ''
    s_vector = np.full(6, -1)
    s_vector[0] = 1
    cur_classification = 1

    for j in range(6):
        if s_vector[j] == -1:
            cur_classification += 1
            s_vector[j] = cur_classification

        for k in range(j + 1, 6):
            if num[k] == num[j]:
                s_vector[k] = s_vector[j]

        Spatial += str(s_vector[j])

    # 计算时间属性 Temporal
    Temporal = get_t(num, D_heng)

    # 将结果添加到列表中
    pins.append(num)
    spatials.append(Spatial)
    temporal.append(Temporal)

    # 每 10000 次打印一次进度，减少 I/O 操作
    if i % 100000 == 0:
        print(f'目前已到达 {i}')

# 在循环结束后创建 DataFrame
df = pd.DataFrame({"PIN": pins, "Spatial": spatials, "Temporal": temporal})

# 将结果保存到 CSV 文件中
df.to_csv('attack/table/6digit_pin.csv', index=False)
