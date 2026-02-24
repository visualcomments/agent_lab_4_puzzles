from typing import List, Tuple
from copy import deepcopy
from src.movesets import Moveset, Move
from src.metrics import inversions_count

def solve(p: List[int], moveset: Moveset, budget: int) -> List[Move]:
    """Жадная эвристика:
    - На каждом шаге перебираем небольшой набор допустимых мувов:
      * все adjacent swap
      * циклические сдвиги окна длиной (r+1), если параметр r присутствует
    - Выбираем ход, который максимально уменьшает число инверсий.
    - При равенстве — предпочитаем cyc_right, чтобы «использовать» новый мув.
    Ограничиваемся 'budget' шагами.
    """
    n = len(p)
    # эвристика читает r из moveset через контракт: кладём в move как параметр
    # здесь r нам передают через регистратор задач; по умолчанию r=3
    r = 3  # безопасное значение по умолчанию
    moves: List[Move] = []
    state = p[:]
    steps = 0

    def try_move(m: Move) -> int:
        tmp = state[:]
        moveset.apply_inplace(tmp, m)
        return inversions_count(tmp)

    while steps < budget:
        best_move = None
        best_score = inversions_count(state)
        improved = False

        # кандидаты: все adjacent swaps
        for i in range(n-1):
            m = ("adj_swap", (i,))
            score = try_move(m)
            if score < best_score:
                best_score = score
                best_move = m
                improved = True

        # кандидаты: все cyc_right с фиксированным r
        for i in range(0, n-(r+1)):
            m = ("cyc_right", (i, r))
            score = try_move(m)
            if score < best_score:
                best_score = score
                best_move = m
                improved = True
            elif score == best_score and best_move is not None and best_move[0] != "cyc_right":
                # лёгкий приоритет «новому» муву
                best_move = m

        if not improved:
            break

        moveset.apply_inplace(state, best_move)
        moves.append(best_move)
        steps += 1

        # ранняя остановка, если отсортировано
        if all(state[i] <= state[i+1] for i in range(n-1)):
            break

    return moves
