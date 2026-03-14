import argparse
from src.registry import TASKS, CANDIDATES
from src.evaluator import evaluate_candidate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="cyclic-coxeter-n16-r3")
    parser.add_argument("--splits", nargs="+", default=["fast"]
                        , help="Какие сплиты запускать: fast, control")
    parser.add_argument("--candidates", nargs="+", default=["baseline", "heuristic1"]
                        , help="Список кандидатов по именам реестра")
    args = parser.parse_args()

    if args.task not in TASKS:
        raise SystemExit(f"Неизвестная задача: {args.task}. Доступны: {list(TASKS)}")
    task = TASKS[args.task]()

    # фильтр сплитов
    task.splits = {k: v for k, v in task.splits.items() if k in set(args.splits)}

    for name in args.candidates:
        if name not in CANDIDATES:
            raise SystemExit(f"Неизвестный кандидат: {name}. Доступны: {list(CANDIDATES)}")
        print("\n=== Candidate:", name, "===")
        fn = CANDIDATES[name]
        r = task.moveset_params.get("r", 3)
        res = evaluate_candidate(task, fn, r=r)
        for split, metrics in res.items():
            print(f"[{split}]", {k: round(v, 4) for k, v in metrics.items()})

if __name__ == "__main__":
    main()
