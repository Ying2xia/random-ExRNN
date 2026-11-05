# ExtremileFed 复现代码

本目录对应当前 TKDE 投稿稿件。核心算法代码保留原样，另外整理了可直接读取固定配置的入口，不依赖作者电脑上的历史工作目录。

## 安装和检查

在本目录打开终端，安装 `requirements.txt` 中的依赖，然后执行：

```sh
PYTHONPATH=code python -B -m unittest discover -s code/tests
python -B code/reproduce_main.py --smoke --run
python -B code/run_fair_baseline_confirmation.py --smoke --workers 2
```

已记录的运行环境是 Python 3.9.25、PyTorch 2.8.0；具体版本见 `docs/tested_environment.json`。入口使用 POSIX 进程锁，在 macOS 检查；Linux 可使用相同的启动方式。

## 正式实验入口

| 实验 | 入口 | 固定配置 |
|---|---|---|
| 12 种设置的主模拟 | `code/reproduce_main.py` | `configs/main/` |
| 五种代表设置的敏感性与组件实验 | `code/reproduce_supplement.py` | `configs/supplementary/` |
| 同 Host 初值的增强基线比较 | `code/run_fair_baseline_confirmation.py` | `configs/main/`、`configs/fair/` |
| 保险与空气质量应用 | `code/reproduce_real.py` | `configs/real/` |

主模拟和补充模拟默认只列出计划，加 `--run` 才开始拟合。公平基线入口默认执行完整开发、选参和确认流程，查看方案用 `--dry-run`。实际复现命令和分阶段用法见英文 `README.md`。

主模拟固定一层隐藏层。补充实验使用各自固定的参考设置，因此配置单独保存。新入口保持原数据种子、网络种子、拟合实现和评价口径；补充参考拟合重新计算，不依赖旧结果缓存。真实数据只使用已固定设置评估，选参代码另行保留在原应用入口中。

## 哪些文件需要上传

上传本目录全部文件，包括 `code/`、`configs/`、`published_results/`、`data/` 的说明与校验值、`docs/`、依赖说明、引用信息及 `.gitignore`。

`published_results/` 是从正式记录提取的逐重复指标和小体积汇总，足以检查论文比较并重算均值与标准误，不包含逐条观测数据、训练轨迹或模型权重。执行 `python -B code/summarize_results.py` 可重算主模拟和补充模拟汇总，执行 `python -B code/plot_results.py` 可生成模拟 RMSE 图和真实数据剖面图。

原始数据从官方来源单独准备，详见 `data/README.md`。运行产生的 `outputs/`、`code/results/`、模型检查点、预测数组、日志和 Python 缓存已由 `.gitignore` 排除。原工程中的论文、参考文献 PDF、历史备份及未入稿的异质性探索没有放入本目录。
