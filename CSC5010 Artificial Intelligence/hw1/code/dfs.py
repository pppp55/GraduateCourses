from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


State = Tuple[int, ...]
GOAL_STATE: State = (1, 2, 3, 8, 0, 4, 7, 6, 5)
FIXED_START_STATE: State = (2, 8, 3, 1, 6, 4, 7, 0, 5)
STUDENT_ID = 225040065
REPORT_PATH = Path(__file__).with_name("search_report.txt")


@dataclass
class SearchResult:
    path: List[State]
    moves: int
    expanded: int


def str_to_state(s: str) -> State:
    return tuple(int(ch) for ch in s)


def state_to_str(state: State) -> str:
    return "".join(map(str, state))


def inversion_parity(state: State) -> int:
    nums = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                inversions += 1
    return inversions % 2


def is_reachable(start: State, goal: State) -> bool:
    return inversion_parity(start) == inversion_parity(goal)


def neighbors(state: State) -> Iterable[State]:
    i = state.index(0)
    r, c = divmod(i, 3)
    candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    for nr, nc in candidates:
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            nxt = list(state)
            nxt[i], nxt[j] = nxt[j], nxt[i]
            yield tuple(nxt)


def reconstruct_path(parent: Dict[State, Optional[State]], end: State) -> List[State]:
    path: List[State] = []
    cur: Optional[State] = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def dfs(start: State, goal: State) -> Optional[SearchResult]:
    if start == goal:
        return SearchResult(path=[start], moves=0, expanded=0)

    stack: List[State] = [start]
    parent: Dict[State, Optional[State]] = {start: None}
    visited = {start}
    expanded = 0

    while stack:
        cur = stack.pop()
        expanded += 1

        if cur == goal:
            path = reconstruct_path(parent, cur)
            return SearchResult(path=path, moves=len(path) - 1, expanded=expanded)

        nxt_states = list(neighbors(cur))
        # Reverse push order to keep traversal deterministic.
        for nxt in reversed(nxt_states):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = cur
                stack.append(nxt)

    return None


def manhattan(state: State, goal_pos: Dict[int, Tuple[int, int]]) -> int:
    distance = 0
    for idx, val in enumerate(state):
        if val == 0:
            continue
        r, c = divmod(idx, 3)
        gr, gc = goal_pos[val]
        distance += abs(r - gr) + abs(c - gc)
    return distance


def a_star(start: State, goal: State) -> Optional[SearchResult]:
    if start == goal:
        return SearchResult(path=[start], moves=0, expanded=0)

    goal_pos = {v: divmod(i, 3) for i, v in enumerate(goal)}
    open_heap: List[Tuple[int, int, State]] = []
    g_cost: Dict[State, int] = {start: 0}
    parent: Dict[State, Optional[State]] = {start: None}
    counter = 0
    heappush(open_heap, (manhattan(start, goal_pos), counter, start))
    expanded = 0

    while open_heap:
        _, _, cur = heappop(open_heap)
        expanded += 1

        if cur == goal:
            path = reconstruct_path(parent, cur)
            return SearchResult(path=path, moves=len(path) - 1, expanded=expanded)

        cur_g = g_cost[cur]
        for nxt in neighbors(cur):
            nxt_g = cur_g + 1
            if nxt not in g_cost or nxt_g < g_cost[nxt]:
                g_cost[nxt] = nxt_g
                parent[nxt] = cur
                counter += 1
                f = nxt_g + manhattan(nxt, goal_pos)
                heappush(open_heap, (f, counter, nxt))

    return None


def random_state_pair(seed: int) -> Tuple[State, State]:
    rng = random.Random(seed)
    digits = list(range(9))
    rng.shuffle(digits)
    start = tuple(digits)
    rng.shuffle(digits)
    goal = tuple(digits)
    return start, goal


def swap_neighboring_numbers(state: State) -> State:
    # Swap the first neighboring pair that does not contain 0.
    # This flips inversion parity for 8-puzzle reachability.
    arr = list(state)
    for i in range(len(arr) - 1):
        if arr[i] != 0 and arr[i + 1] != 0:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            return tuple(arr)
    raise ValueError("No valid neighboring pair to swap.")


def print_result(
    title: str, start: State, goal: State, result: Optional[SearchResult]
) -> None:
    print(f"\n=== {title} ===")
    print(f"Start: {state_to_str(start)}")
    print(f"Goal : {state_to_str(goal)}")

    if result is None:
        print("No solution found.")
        write_report(title, start, goal, None)
        return

    print(f"Moves      : {result.moves}")
    path_preview = preview_path(result.path)
    print(f"Path       : {path_preview}")
    write_report(title, start, goal, result)


def path_to_states_line(path: List[State]) -> str:
    return " -> ".join(state_to_str(state) for state in path)


def preview_path(path: List[State], max_states: int = 6) -> str:
    states = [state_to_str(s) for s in path]
    if len(states) <= max_states:
        return " -> ".join(states)
    head = " -> ".join(states[:3])
    tail = " -> ".join(states[-3:])
    return f"{head} -> ... -> {tail}"


def write_report(
    title: str, start: State, goal: State, result: Optional[SearchResult]
) -> None:
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write(f"=== {title} ===\n")
        report.write(f"Start: {state_to_str(start)}\n")
        report.write(f"Goal : {state_to_str(goal)}\n")
        if result is None:
            report.write("No solution found.\n\n")
            return
        report.write(f"Moves: {result.moves}\n")
        report.write("Path:\n")
        report.write(path_to_states_line(result.path) + "\n\n")


def run_fixed_tasks() -> None:
    REPORT_PATH.write_text("8-puzzle Search Report\n\n", encoding="utf-8")
    start = FIXED_START_STATE
    goal = GOAL_STATE

    # A1: single-person odd ID -> DFS
    if is_reachable(start, goal):
        dfs_result = dfs(start, goal)
        print_result("A1: DFS on fixed puzzle", start, goal, dfs_result)
    else:
        print_result("A1: DFS on fixed puzzle", start, goal, None)

    # B1: A* on the same puzzle
    if is_reachable(start, goal):
        astar_result = a_star(start, goal)
        print_result("B1: A* on fixed puzzle", start, goal, astar_result)
    else:
        print_result("B1: A* on fixed puzzle", start, goal, None)

    # C1: parity test by swapping first two numbers in start state
    parity_start = list(start)
    parity_start[0], parity_start[1] = parity_start[1], parity_start[0]
    parity_start_state = tuple(parity_start)
    print(f"\n=== C1: Parity test ===")
    print(f"Original start parity: {inversion_parity(start)}")
    print(f"Swapped  start parity: {inversion_parity(parity_start_state)}")
    print(f"Goal     parity: {inversion_parity(goal)}")
    print(f"Reachable after swap? {is_reachable(parity_start_state, goal)}")
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write("=== C1: Parity test ===\n")
        report.write(f"Original start parity: {inversion_parity(start)}\n")
        report.write(f"Swapped start parity : {inversion_parity(parity_start_state)}\n")
        report.write(f"Goal parity          : {inversion_parity(goal)}\n")
        report.write(
            f"Reachable after swap?: {is_reachable(parity_start_state, goal)}\n\n"
        )


def run_random_tasks(student_id: int) -> None:
    # A2/B2: random start and goal with student ID as seed
    start, goal = random_state_pair(student_id)

    print(f"\n=== A2/B2: Random puzzle (seed={student_id}) ===")
    print(f"Start: {state_to_str(start)}")
    print(f"Goal : {state_to_str(goal)}")
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write(f"=== A2/B2: Random puzzle (seed={student_id}) ===\n")
        report.write(f"Start: {state_to_str(start)}\n")
        report.write(f"Goal : {state_to_str(goal)}\n")

    if not is_reachable(start, goal):
        print("These two states have different parity. Goal is unreachable.")
        start = swap_neighboring_numbers(start)
        print("Swap neighboring numbers in start state and retry.")
        print(f"New start: {state_to_str(start)}")
        with REPORT_PATH.open("a", encoding="utf-8") as report:
            report.write("Parity mismatch: True\n")
            report.write(f"Adjusted start: {state_to_str(start)}\n")
    else:
        with REPORT_PATH.open("a", encoding="utf-8") as report:
            report.write("Parity mismatch: False\n")

    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write("\n")

    dfs_result = dfs(start, goal)
    astar_result = a_star(start, goal)
    print_result("A2: DFS on random puzzle", start, goal, dfs_result)
    print_result("B2: A* on random puzzle", start, goal, astar_result)


if __name__ == "__main__":
    run_fixed_tasks()
    run_random_tasks(STUDENT_ID)
    print(f"\nReport written to: {REPORT_PATH}")
