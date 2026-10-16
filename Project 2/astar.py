"""Project 2：单文件 A* 栅格路径规划（Python 3.10+）。

安装绘图依赖：python -m pip install matplotlib
运行示例：python astar.py
默认在终端展示四、八邻域；缺少绘图库时跳过图片，保留搜索结果。
只输出文本结果、不生成图片：python astar.py --no-plot
自定义输入：python astar.py --input map.json --movement 8 --heuristic octile
坐标采用 (行, 列)，0 表示空地，1 表示障碍物。
"""

import argparse
from dataclasses import dataclass
import heapq
import itertools
import json
import math
from pathlib import Path
import sys
from time import perf_counter


HEURISTICS = {
    4: ('zero', 'manhattan', 'euclidean', 'chebyshev'),
    8: ('zero', 'octile', 'euclidean', 'chebyshev'),
}
NAMES = {'zero': '零启发式', 'manhattan': '曼哈顿', 'euclidean': '欧氏',
         'chebyshev': '切比雪夫', 'octile': 'Octile'}


@dataclass
class SearchResult:
    """无路时 path=[]、cost=None；expanded 包含有效出堆的节点（含目标）。"""
    path: list
    cost: float | None
    expanded: list
    elapsed_ms: float


def validate(grid, start, goal):
    """校验普通列表／元组及内置 int；不接受 bool 或 NumPy 整数。"""
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError('地图必须是非空二维数组。')
    if not isinstance(grid[0], (list, tuple)) or not grid[0]:
        raise ValueError('地图行不能为空。')
    width = len(grid[0])
    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError('地图必须是矩形，各行长度一致。')
        if any(type(cell) is not int or cell not in (0, 1) for cell in row):
            raise ValueError('地图只能包含整数 0 和 1。')
    for name, point in (('起点', start), ('终点', goal)):
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(value) is not int for value in point)):
            raise ValueError(f'{name}必须是两个整数组成的坐标。')
        r, c = point
        if not (0 <= r < len(grid) and 0 <= c < width):
            raise ValueError(f'{name}超出地图范围。')
        if grid[r][c] == 1:
            raise ValueError(f'{name}不能位于障碍物上。')


def estimate(point, goal, name):
    """计算 h；允许的函数均为对应移动模型的代价下界。"""
    dx, dy = abs(point[0] - goal[0]), abs(point[1] - goal[1])
    if name == 'zero':
        return 0.0
    if name == 'manhattan':
        return dx + dy
    if name == 'euclidean':
        return math.hypot(dx, dy)
    if name == 'chebyshev':
        return max(dx, dy)
    if name == 'octile':
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
    raise ValueError(f'未知启发式函数：{name}')


def neighbors(grid, point, movement):
    """固定方向顺序；对角移动必须保证两侧正交格均为空地。"""
    r, c = point
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    if movement == 8:
        directions += [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
            continue
        if grid[nr][nc] == 1:
            continue
        diagonal = dr != 0 and dc != 0
        if diagonal and (grid[r][nc] == 1 or grid[nr][c] == 1):
            continue
        yield (nr, nc), math.sqrt(2) if diagonal else 1.0


def astar(grid, start, goal, movement=4, heuristic=None, tie_break='low_h'):
    """返回最短路径。low_h：f 相同时优先小 h；fifo：优先早入队。

    不设置永久封闭集合，出现更小 g 时允许重新入堆。
    堆条目携带入队时的 g，出堆时剔除过期条目。
    """
    validate(grid, start, goal)
    start, goal = tuple(start), tuple(goal)
    if type(movement) is not int or movement not in HEURISTICS:
        raise ValueError('拓展方式只能是整数 4 或 8。')
    heuristic = heuristic if heuristic is not None else ('manhattan' if movement == 4 else 'octile')
    if heuristic not in HEURISTICS[movement]:
        raise ValueError(f'{movement} 邻域支持：{", ".join(HEURISTICS[movement])}。')
    if tie_break not in ('low_h', 'fifo'):
        raise ValueError('同优先级策略只能是 low_h 或 fifo。')

    begin = perf_counter()
    serial = itertools.count()
    h0 = estimate(start, goal, heuristic)
    # 排序字段：f、次级优先级、入队序号；坐标不参与优先级决策。
    queue = [(h0, h0 if tie_break == 'low_h' else 0, next(serial), 0.0, start)]
    g_score, parent, expanded = {start: 0.0}, {}, []
    while queue:
        _, _, _, queued_g, current = heapq.heappop(queue)
        if queued_g != g_score[current]:
            continue
        expanded.append(current)
        # 目标有效出堆时结束，不能在第一次遇到目标邻居时结束。
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return SearchResult(path, queued_g, expanded, (perf_counter()-begin)*1000)
        for nxt, step_cost in neighbors(grid, current, movement):
            new_g = queued_g + step_cost
            if new_g < g_score.get(nxt, math.inf):
                g_score[nxt], parent[nxt] = new_g, current
                h = estimate(nxt, goal, heuristic)
                heapq.heappush(queue, (new_g+h, h if tie_break == 'low_h' else 0,
                                      next(serial), new_g, nxt))
    return SearchResult([], None, expanded, (perf_counter()-begin)*1000)


def demo_map():
    """15×20 地图：三道有缺口的墙迫使路径绕行，可直接修改数组。"""
    grid = [[0]*20 for _ in range(15)]
    for column, rows, gap in [(5, range(12), 4), (10, range(3, 15), 10),
                              (15, range(12), 6)]:
        for row in rows:
            if row != gap:
                grid[row][column] = 1
    return grid, (1, 1), (13, 18)


def draw_results(grid, start, goal, selected, movement, heuristic, tie_break, output):
    """一张图展示路径和策略对比；每个指标来自实际 A* 调用。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    other_movement = 8 if movement == 4 else 4
    other_h = 'octile' if other_movement == 8 else 'manhattan'
    # 两幅路径图使用相同的同优先级策略，另一邻域采用其默认启发式。
    other = astar(grid, start, goal, other_movement, other_h, tie_break)
    rows, cols = len(grid), len(grid[0])
    fig = plt.figure(figsize=(13, 10), facecolor='white')
    layout = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.22, wspace=0.12)
    fig.suptitle('A* 栅格路径规划｜实际运行结果', fontsize=20, fontweight='bold', y=0.975)
    colors = ListedColormap(['#F1F5F9', '#263445', '#D6E8FB'])
    for index, (result, moves, hname, tie) in enumerate([
            (selected, movement, heuristic, tie_break), (other, other_movement, other_h, tie_break)]):
        ax = fig.add_subplot(layout[0, index])
        cells = [row[:] for row in grid]
        for r, c in result.expanded:
            cells[r][c] = 2
        ax.imshow(cells, cmap=colors, vmin=0, vmax=2, interpolation='nearest')
        if result.path:
            ax.plot([c for r, c in result.path], [r for r, c in result.path],
                    color='#ED8D23', linewidth=2.8, zorder=3)
        for point, color, label in [(start, '#16916C', 'S'), (goal, '#D64157', 'G')]:
            ax.scatter(point[1], point[0], c=color, s=180, edgecolors='white', zorder=4)
            ax.text(point[1], point[0], label, color='white', ha='center', va='center',
                    fontsize=9, fontweight='bold', zorder=5)
        cost = f'{result.cost:.3f}' if result.cost is not None else '不可达'
        ax.set_title(f'{moves} 邻域 · {NAMES[hname]} · {tie}\n'
                     f'代价 {cost}  |  步数 {max(len(result.path)-1, 0)}  |  扩展 {len(result.expanded)}',
                     fontsize=11, pad=10)
        ax.set_xlabel('列 col（从 0 开始）')
        ax.set_ylabel('行 row（从 0 开始）')
        ax.set_xticks(range(0, cols, max(1, cols//10)))
        ax.set_yticks(range(0, rows, max(1, rows//10)))
        if max(rows, cols) <= 60:
            ax.set_xticks([n-0.5 for n in range(cols+1)], minor=True)
            ax.set_yticks([n-0.5 for n in range(rows+1)], minor=True)
            ax.grid(which='minor', color='white', linewidth=0.6)
            ax.tick_params(which='minor', bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend = [Patch(color='#263445', label='障碍物'), Patch(color='#D6E8FB', label='已扩展格'),
              Line2D([0], [0], color='#ED8D23', linewidth=3, label='最短路径'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor='#16916C', label='起点 S'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor='#D64157', label='终点 G')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.947), ncol=5, frameon=False)

    ax = fig.add_subplot(layout[1, :])
    ax.axis('off')
    ax.set_title('同一地图的策略对比（按移动模型分别比较）', fontsize=14, pad=8)
    data = []
    for moves in (4, 8):
        for hname in HEURISTICS[moves]:
            low = astar(grid, start, goal, moves, hname, 'low_h')
            fifo = astar(grid, start, goal, moves, hname, 'fifo')
            if low.cost is None or fifo.cost is None:
                assert low.cost is None and fifo.cost is None
            else:
                assert math.isclose(low.cost, fifo.cost, rel_tol=1e-9)
            data.append([f'{moves} 邻域', NAMES[hname],
                         f'{low.cost:.3f}' if low.cost is not None else '不可达',
                         len(low.expanded), len(fifo.expanded)])
    table = ax.table(cellText=data,
                     colLabels=['拓展方式', '启发式函数', '最短代价', '扩展数：小 h 优先', '扩展数：入队顺序'],
                     cellLoc='center', colLoc='center', bbox=[0.03, 0.09, 0.94, 0.83])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#17364A' if r == 0 else ('#EDF3F8' if r % 2 else '#F8FAFC'))
        if r == 0:
            cell.set_text_props(color='white', weight='bold')
    ax.text(0.5, 0, '正交代价 1；对角代价 √2；禁止穿障碍物角。扩展数含起点和目标，零启发式等价于 Dijkstra。',
            ha='center', va='bottom', transform=ax.transAxes, fontsize=10, color='#526476')
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.06, right=0.97)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='A* 最短路径搜索；坐标为 (行, 列)。')
    parser.add_argument('--input', type=Path, help='JSON 文件，包含 grid、start、goal')
    parser.add_argument('--movement', type=int, choices=(4, 8), default=None,
                        help='指定 4 或 8；未指定邻域和启发式时展示两种邻域；仅指定启发式时默认 4')
    parser.add_argument('--heuristic', choices=sorted(set(sum(HEURISTICS.values(), ()))), default=None,
                        help='启发式函数；octile 需配合 --movement 8，manhattan 仅用于 4 邻域')
    parser.add_argument('--tie', choices=('low_h', 'fifo'), default='low_h')
    parser.add_argument('--no-plot', action='store_true', help='只输出文本结果，不加载绘图库、不生成图片')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('运行结果.png'))
    args = parser.parse_args()
    try:
        # 在读取、显示地图和执行搜索前，先检查参数组合及图片输出路径。
        movements = (4, 8) if args.movement is None and args.heuristic is None else (args.movement or 4,)
        if args.heuristic is not None and args.heuristic not in HEURISTICS[movements[0]]:
            if args.heuristic == 'octile':
                raise ValueError('octile 仅用于八邻域，请同时指定 --movement 8。')
            raise ValueError(f'启发式 {args.heuristic} 不支持 {movements[0]} 邻域；'
                             f'可选：{", ".join(HEURISTICS[movements[0]])}。')
        if not args.no_plot:
            if not args.output.parent.is_dir():
                raise ValueError('图片输出目录不存在，请先创建目录。')
            if args.output.is_dir():
                raise ValueError('图片输出路径是目录，请指定图片文件名，例如 result.png。')
        if args.input:
            content = json.loads(args.input.read_text(encoding='utf-8-sig'))
            if not isinstance(content, dict) or not all(k in content for k in ('grid', 'start', 'goal')):
                raise ValueError('输入 JSON 必须包含 grid、start、goal。')
            grid, start, goal = content['grid'], content['start'], content['goal']
        else:
            grid, start, goal = demo_map()
        # 必须先校验，才能安全读取尺寸和转换坐标；astar 也独立校验其调用参数。
        validate(grid, start, goal)
        print(f'地图：{len(grid)}×{len(grid[0])}；起点：{tuple(start)}；终点：{tuple(goal)}')
        # 默认展示两个模型；显式指定参数时仅搜索所选模型。
        selected = None
        for movement in movements:
            hname = args.heuristic if args.heuristic is not None else ('manhattan' if movement == 4 else 'octile')
            result = astar(grid, start, goal, movement, hname, args.tie)
            if selected is None:
                selected = result, movement, hname
            print(f'\n拓展：{movement} 邻域；启发式：{hname}；同优先级策略：{args.tie}')
            print('状态：找到最短路径' if result.path else '状态：无路可达')
            print(f'坐标序列：{result.path}')
            print(f'路径步数：{max(len(result.path)-1, 0)}；总代价：{result.cost}')
            print(f'扩展节点数：{len(result.expanded)}；搜索耗时：{result.elapsed_ms:.3f} ms')
        if not args.no_plot:
            result, movement, hname = selected
            try:
                draw_results(grid, tuple(start), tuple(goal), result, movement, hname, args.tie, args.output)
            except ImportError as exc:
                print(f'\n搜索已完成，缺少或无法加载绘图依赖，已跳过图片生成：{exc}')
                print(f'当前 Python：{sys.executable}')
                print('请在 VS Code 的 PowerShell 终端执行以下命令，再重新运行：')
                print(f'& "{sys.executable}" -m pip install matplotlib')
                return
            print(f'运行结果图：{args.output.resolve()}')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
