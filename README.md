# Daily Shortlist Workstation

一个本地运行的美股短线分析面板，用来把行情、新闻和简单的规则模型整理成当天的候选名单，帮助我更快判断“今天该先看谁”。

这个项目不是自动交易机器人，也不会直接下单。它更像一个盘前 / 盘中研究工具：先把值得关注的股票筛出来，再给出价格区间、结构判断和风险提示，最后由我自己在 IBKR 手动执行。

## 我为什么做这个项目

我一直对短线市场里“消息如何转化为价格波动”这件事很感兴趣。大学里学过一些经济相关内容，后来在看美股的时候，越来越明显地感觉到：

- 短时间内波动大的股票，往往先出现在异动榜、新闻流和热门关键词里
- 单靠看新闻太慢，单靠看价格又容易没有上下文
- 真正有用的，不是一个“神预测模型”，而是一套能快速整理候选、说明理由、提示风险的分析流程

所以我做了这套系统，把几个最实用的环节串起来：

- 从 Alpaca 拉市场数据
- 从 Alpaca News 和 SEC RSS 拉催化信息
- 做一层解释性比较强的特征和评分
- 给出 shortlist、结构标签和关键价格区间
- 把结果存下来，方便后面复盘和继续验证

## 这个项目现在能做什么

### 1. 生成当日 shortlist

系统每天会从几类信息里先构建候选池：

- Alpaca movers / 异动股
- Alpaca News / SEC RSS 里最近出现的股票
- 一组核心观察名单，防止数据源很安静时完全没有候选

然后再补齐这些股票的市场上下文，包括：

- snapshot
- latest quote
- latest bar
- intraday bars
- daily bars

最后为每只股票生成一组特征，做筛选和排序。

### 2. 解释为什么这只股票会入选

每只股票都会有一套比较透明的解释字段，而不是只有一个黑箱分数。当前会输出：

- `news_score`
- `setup_score`
- `liquidity_score`
- `risk_score`
- `total_score`
- `status_tag`
- `tradeable`
- `tradeable_reason`
- `not_tradeable_reason`
- `selection_reason`
- `risk_note`
- `action_note`

也就是说，不只是告诉你“它排第几”，还会说明：

- 它为什么进候选池
- 结构更像 breakout 还是 pullback
- 现在适不适合主动交易
- 最主要的风险在哪里

### 3. 给出短线观察区间

系统不会直接给出“买入建议”或“保证上涨目标”，而是给一组更适合盘中观察的价格带：

- `Base Range`
- `Breakout Zone`
- `Pullback Zone`
- `Invalidation`

这些区间是根据当前价、VWAP、近端波动和最近支撑/阻力算出来的，用来帮助判断：

- 现在是在合理观察区，还是已经太延伸
- 更适合等突破，还是等回踩
- 失效位在哪里

### 4. 本地仪表盘查看结果

项目用 Streamlit 做了一个本地工作台，主要页面包括：

- `Dashboard`
- `Shortlist`
- `Replay`
- `History`
- `Activity`
- `Files`

我平时主要看的是：

- 顶部状态栏：当前是 live 还是 mock，数据源是否连通
- shortlist 主表：今天先盯哪些股票
- 单票详情：价格区间、quote、bars、新闻、风险说明
- replay：历史候选的简单命中情况和后续研究入口

### 5. 记录运行结果，方便复盘

每次运行都会写出：

- 最新 shortlist CSV
- HTML 报告
- `run_status.json`
- `activity_log.json`
- `latest_context.json`
- `shortlist.db`

其中 `SQLite` 会保存：

- 候选快照
- 新闻事件
- 区间结果
- 模型运行记录

这部分是为了以后继续做 replay、策略验证和 walk-forward 检查。

## 从数据到候选，整个流程是怎么跑的

这是这个项目最核心的一条链路：

1. 读取 `.env` 和运行参数  
2. 探测 Alpaca / News / SEC RSS 当前是否可用  
3. 拉取 movers，作为第一层候选  
4. 拉取 Alpaca News 和 SEC RSS，补充有催化的股票  
5. 合并成候选 seed symbols  
6. 为这些 symbol 拉 snapshot、quote、bars 等市场数据  
7. 生成特征  
8. 做规则评分  
9. 计算区间模型  
10. 应用硬过滤和 tradeable 规则  
11. 生成 shortlist  
12. 写入 CSV / HTML / SQLite / 本地状态文件  
13. 在 Streamlit 页面里展示

简单说，这套程序不是“先预测所有股票”，而是：

**先把值得看的股票找出来，再判断哪些更值得交易。**

## 评分逻辑

目前主评分仍然是可解释的规则模型，核心考虑的是四件事：

- 催化够不够新
- 流动性够不够好
- 当前结构够不够清晰
- 风险是不是太高

总分大致由下面几部分组成：

- catalyst / news
- setup
- liquidity
- risk
- ml（如果模型层可用）

如果某只股票满足这些问题中的任意几个，它会被降权或者直接排除：

- spread 太宽
- 成交额太低
- 相对量能不够
- 没有明确催化，只有随机波动
- 已经明显延伸，不适合继续追

## 目录结构

```text
daily_shortlist_engine/
├── app.py                  # Streamlit 本地工作台
├── main.py                 # 单次运行入口
├── requirements.txt
├── .env.example
├── data/                   # 运行生成的数据文件（默认不提交）
├── outputs/                # 导出的 CSV / HTML（默认不提交）
└── src/
    ├── alpaca_client.py
    ├── project_env.py
    ├── config/
    │   └── settings.py
    ├── data/
    │   ├── alpaca_market.py
    │   ├── alpaca_news.py
    │   ├── sec_rss.py
    │   └── dataset_builder.py
    ├── engine/
    │   ├── features.py
    │   ├── scoring.py
    │   ├── filters.py
    │   ├── labels.py
    │   ├── range_model.py
    │   ├── regime.py
    │   ├── ml_features.py
    │   └── ml_targets.py
    ├── models/
    │   ├── lightgbm_ranker.py
    │   ├── lightgbm_classifier.py
    │   ├── lightgbm_regressor.py
    │   ├── model_registry.py
    │   └── model_service.py
    ├── storage/
    │   ├── db.py
    │   └── models.py
    ├── ui/
    │   ├── dashboard.py
    │   ├── detail_panel.py
    │   ├── ml_panel.py
    │   └── replay_page.py
    └── backtest/
        ├── walkforward.py
        ├── evaluation.py
        └── strategy_templates.py
```

## 当前技术栈

- Python
- Streamlit
- pandas
- requests
- SQLite
- Jinja2
- scikit-learn
- LightGBM

## 如何在本地运行

### 1. 创建并配置环境变量

```bash
cp .env.example .env
```

最少需要填这些：

```env
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://data.alpaca.markets
SEC_USER_AGENT=Your Name your@email.com
MOCK_MODE=false
```

如果 `SEC_USER_AGENT` 还是占位值，SEC RSS 会被标记为 unavailable，这是预期行为。

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果是 macOS 且 `lightgbm` 安装后导入失败，通常需要额外安装 `libomp`。

### 3. 运行数据流程

```bash
python main.py
```

### 4. 打开本地页面

```bash
streamlit run app.py --server.port 8503
```

浏览器打开：

- [http://localhost:8503](http://localhost:8503)

## 当前状态

目前这套系统已经可以完成下面这些事情：

- 连接 Alpaca Market Data API
- 获取 movers、snapshots、latest quotes、latest bars、historical bars
- 获取 Alpaca News
- 获取 SEC RSS（前提是 `SEC_USER_AGENT` 合规）
- 生成 shortlist
- 计算规则分数和结构标签
- 计算 Base / Breakout / Pullback / Invalidation
- 保存到 SQLite 供后续 replay 使用
- 在本地仪表盘里展示运行状态、候选表和单票详情

## 还没有完全做完的部分

目前我把这个项目看成一个持续迭代的研究工具，而不是已经完成的产品。还在继续补强的点主要有：

- 更稳的事件主题识别
- 更好的行业 / 主题映射
- 更完整的 replay 指标
- LightGBM 模型在足够样本下的稳定训练和验证
- websocket 驱动的更实时更新
- 更成熟的策略模板研究

换句话说，规则层和本地工作台已经能用，模型层和更深的研究层还在继续打磨。

## 这个项目不做什么

为了避免误解，这里也明确一下它**不做的事**：

- 不自动下单
- 不承诺收益
- 不输出“稳赚”或“无风险”建议
- 不把所有股票都塞进候选
- 不把黑箱模型包装成“神预测”

这个项目的定位一直很明确：  
**它是一个本地运行的短线分析工作台，用来提高观察效率和判断质量，而不是代替交易决策本身。**

## 备注

如果我要把这个项目继续往下推进，我会优先做两件事：

1. 继续提升 shortlist 的可执行性，减少“只能看不能动”的票  
2. 在样本积累足够以后，把 LightGBM 的训练、验证和回放统计真正跑起来

这也是我现在把它放到 GitHub 上的原因：它已经不是一个零散脚本，而是一套有明确结构、可以持续迭代的个人项目。
