import argparse
from recbole.quick_start import run_recbole

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RecBole Benchmarks on custom Book Dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BPR",
        help="你要运行的推荐模型名字，比如 BPR, NCF, LightGCN",
    )

    args = parser.parse_args()

    print(f"正在配置环境并使用 {args.model} 模型跑 book_recipe 数据集...")

    # 调起模型，使用你的自定义配置文件 book_recipe.yaml
    run_recbole(
        model=args.model, dataset="book_recipe", config_file_list=["book_recipe.yaml"]
    )
