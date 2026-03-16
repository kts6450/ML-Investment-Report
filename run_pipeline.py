"""전체 파이프라인 순차 실행 스크립트.

사용법
------
python run_pipeline.py --step all        # 전체 실행
python run_pipeline.py --step collect    # 데이터 수집만
python run_pipeline.py --step train      # 학습만
python run_pipeline.py --step backtest   # 백테스트만
python run_pipeline.py --step serve      # API 서버 실행
"""

import argparse


def step_collect():
    print("\n[1/4] 데이터 수집 시작 ───────────────────")
    from crypto_portfolio.data.collector import collect_all
    collect_all("2023-01-01", "2026-03-14")
    print("[1/4] 데이터 수집 완료\n")


def step_train():
    print("\n[2/4] PPO 학습 시작 ───────────────────────")
    from crypto_portfolio.training.train import train
    model, test_feat, test_price = train()
    print("[2/4] PPO 학습 완료\n")
    return test_feat, test_price


def step_backtest(test_feat=None, test_price=None):
    print("\n[3/4] 백테스트 시작 ───────────────────────")

    if test_feat is None or test_price is None:
        from crypto_portfolio.training.train import prepare_data
        _, _, test_feat, test_price = prepare_data()

    from crypto_portfolio.backtest.backtester import run_backtest, print_metrics, plot_backtest
    result = run_backtest(test_feat, test_price)
    print_metrics(result)
    plot_backtest(result, save_path="backtest_result.png")
    print("[3/4] 백테스트 완료\n")


def step_serve():
    print("\n[4/4] FastAPI 서버 시작 ────────────────────")
    import uvicorn
    from crypto_portfolio.config.settings import API_HOST, API_PORT
    uvicorn.run(
        "crypto_portfolio.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Crypto Portfolio Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "collect", "train", "backtest", "serve"],
        default="all",
        help="실행할 단계 선택",
    )
    args = parser.parse_args()

    if args.step == "collect":
        step_collect()
    elif args.step == "train":
        step_train()
    elif args.step == "backtest":
        step_backtest()
    elif args.step == "serve":
        step_serve()
    elif args.step == "all":
        step_collect()
        test_feat, test_price = step_train()
        step_backtest(test_feat, test_price)
        print("모든 단계 완료! API 서버를 시작하려면:")
        print("  python run_pipeline.py --step serve")


if __name__ == "__main__":
    main()
