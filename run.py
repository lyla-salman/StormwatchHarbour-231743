import argparse
import json
import random
import sys
from collections import Counter, deque


# digit: workers, minimum arrival interval, maximum arrival interval,
#        storms per run, cargo per ship
PARAMETERS = {
    0: (10, 6, 10, 3, 8),
    1: (12, 5, 9, 4, 8),
    2: (14, 6, 10, 4, 9),
    3: (16, 5, 9, 5, 9),
    4: (10, 5, 9, 5, 10),
    5: (12, 6, 10, 3, 10),
    6: (14, 5, 9, 3, 7),
    7: (16, 6, 10, 4, 7),
    8: (12, 4, 8, 5, 8),
    9: (14, 4, 8, 4, 9),
}

WIDTH = 24
HEIGHT = 16
TOTAL_TICKS = 600
WAREHOUSE = (0, 15)
BERTHS = {
    "B1": (3, 0),
    "B2": (9, 0),
    "B3": (15, 0),
    "B4": (21, 0),
}
EXPOSED_CELLS = {BERTHS["B1"], BERTHS["B4"]}

# Initial weights; use development seeds 1-10 before freezing these.
UTILITY_WEIGHTS = {"cargo": 4.0, "distance": 2.0,
                   "crowding": 3.0, "risk": 8.0}


def distance(a, b):
    """Manhattan distance: horizontal and vertical steps only."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Harbour:
    def __init__(self, digit, seed, agent, verbose=False):
        self.rng = random.Random(seed)
        self.agent = agent
        self.verbose = verbose

        workers, low, high, storm_count, cargo = PARAMETERS[digit]
        self.cargo_per_ship = cargo
        self.ships = []
        self.workers = [
            {
                "pos": (i // 3, 15),
                "carrying": False,
                "knocked_out_until": 0,
                "unsafe_until": 0,
                "warning_seen": False,
                "avoid_unsafe": False,
                "target_berth": None,
                "target_ship_id": None,
                "berth_memory": {},
            }
            for i in range(workers)
        ]

        self.cargo_delivered = 0
        self.storm_knockouts = 0
        self.crowding_violations = 0
        self.storm_active = False
        self.storm_warning = False

        # Pre-generate events with one RNG. Agent choices cannot change
        # the weather or arrival sequence used for a matching seed.
        self.storm_starts = []
        while len(self.storm_starts) < storm_count:
            candidate = self.rng.randint(20, 580)
            if all(abs(candidate - start) >= 15
                   for start in self.storm_starts):
                self.storm_starts.append(candidate)

        self.arrival_ticks = set()
        tick = self.rng.randint(low, high)
        while tick < TOTAL_TICKS:
            self.arrival_ticks.add(tick)
            tick += self.rng.randint(low, high)

    def log(self, tick, message):
        if self.verbose:
            print(f"Tick {tick}: {message}", file=sys.stderr)

    def update_weather(self, tick):
        self.storm_warning = any(
            start - 5 <= tick < start for start in self.storm_starts
        )
        self.storm_active = any(
            start <= tick < start + 15 for start in self.storm_starts
        )

        if any(tick == start - 5 for start in self.storm_starts):
            self.log(tick, "Lookout: storm starts in 5 ticks!")

        if tick in self.storm_starts:
            self.log(tick, "Storm begins. B1 and B4 are unsafe.")
            for worker in self.workers:
                if (worker["pos"] in EXPOSED_CELLS
                        and tick >= worker["knocked_out_until"]):
                    worker["carrying"] = False
                    worker["knocked_out_until"] = tick + 20
                    self.storm_knockouts += 1
                    self.log(tick, "Worker knocked out; carried cargo lost.")

        if any(tick == start + 15 for start in self.storm_starts):
            self.log(tick, "Storm ends.")

    def update_ships(self, tick):
        if tick in self.arrival_ticks:
            ship = {
                "id": len(self.ships) + 1,
                "arrival": tick,
                "dock": None,
                "berth": None,
                "left": self.cargo_per_ship,
            }
            self.ships.append(ship)
            self.log(tick, f"Ship {ship['id']} arrives.")

        for berth_name in BERTHS:
            occupied = any(
                ship["berth"] == berth_name and ship["left"] > 0
                for ship in self.ships
            )
            if occupied:
                continue

            waiting_ship = next(
                (ship for ship in self.ships if ship["dock"] is None),
                None,
            )
            if waiting_ship is not None:
                waiting_ship["berth"] = berth_name
                waiting_ship["dock"] = tick
                self.log(
                    tick, f"Ship {waiting_ship['id']} docks at {berth_name}."
                )

    def move_worker(self, worker, target, occupancy):
        """Find a free route using BFS, then take exactly one step."""
        start = worker["pos"]
        if start == target:
            return

        frontier = deque([start])
        previous = {start: None}
        while frontier:
            current = frontier.popleft()
            if current == target:
                break

            x, y = current
            neighbours = [(x, y - 1), (x - 1, y),
                          (x, y + 1), (x + 1, y)]
            for next_cell in neighbours:
                nx, ny = next_cell
                if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT):
                    continue
                if next_cell in previous:
                    continue
                if worker["avoid_unsafe"] and next_cell in EXPOSED_CELLS:
                    continue
                if occupancy[next_cell] >= 3:
                    continue

                previous[next_cell] = current
                frontier.append(next_cell)

        if target not in previous:
            return

        next_step = target
        while previous[next_step] != start:
            next_step = previous[next_step]

        occupancy[start] -= 1
        occupancy[next_step] += 1
        worker["pos"] = next_step

    def update_memory(self, worker, tick, current_ships):
        """Receive warning and berth broadcasts; retain an internal model.

        Workers receive current berth status each tick. Only the Lookout
        exposes a warning; workers do not inspect the storm schedule.
        """
        if self.storm_warning and not worker["warning_seen"]:
            # First warning is five ticks before a 15-tick storm.
            worker["unsafe_until"] = tick + 20
        worker["warning_seen"] = self.storm_warning
        worker["avoid_unsafe"] = tick < worker["unsafe_until"]

        for berth_name in BERTHS:
            ship = current_ships.get(berth_name)
            worker["berth_memory"][berth_name] = {
                "ship_id": ship["id"] if ship is not None else None,
                "observed_at": tick,
            }

    def berth_utility(self, worker, ship):
        """U = 4*C - 2*D - 3*Q - 8*R; all factors are in [0, 1].

        Cargo favours productive berths; distance discourages long trips;
        assignment crowding discourages herding. Risk has the largest
        penalty because a knock-out costs cargo and 20 worker ticks.
        These are starting weights, not experimentally optimal weights.
        """
        berth = ship["berth"]
        cargo = ship["left"] / self.cargo_per_ship
        travel = distance(worker["pos"], BERTHS[berth]) / (WIDTH + HEIGHT - 2)
        assigned = sum(
            other is not worker
            and other["target_berth"] == berth
            and other["target_ship_id"] == ship["id"]
            for other in self.workers
        )
        crowding = assigned / max(1, len(self.workers) - 1)
        risk = float(worker["avoid_unsafe"] and berth in ("B1", "B4"))
        return (UTILITY_WEIGHTS["cargo"] * cargo
                - UTILITY_WEIGHTS["distance"] * travel
                - UTILITY_WEIGHTS["crowding"] * crowding
                - UTILITY_WEIGHTS["risk"] * risk)

    def update_workers(self, tick):
        if self.agent == "baseline":
            return

        occupancy = Counter(worker["pos"] for worker in self.workers)
        current_ships = {
            ship["berth"]: ship
            for ship in self.ships
            if ship["dock"] is not None and ship["left"] > 0
        }

        # Rotate turn order so the same worker is not always first.
        for offset in range(len(self.workers)):
            worker = self.workers[(offset + tick) % len(self.workers)]
            if self.agent in ("model", "utility"):
                self.update_memory(worker, tick, current_ships)

            if tick < worker["knocked_out_until"]:
                continue

            if worker["carrying"]:
                if worker["pos"] == WAREHOUSE:
                    self.cargo_delivered += 1
                    worker["carrying"] = False
                    continue
                self.move_worker(worker, WAREHOUSE, occupancy)
                continue

            available_ships = [
                ship for ship in current_ships.values() if ship["left"] > 0
            ]

            if self.agent == "model" and worker["avoid_unsafe"]:
                available_ships = [
                    ship for ship in available_ships
                    if ship["berth"] not in ("B1", "B4")
                ]

            if not available_ships:
                worker["target_berth"] = None
                worker["target_ship_id"] = None
                if worker["avoid_unsafe"]:
                    self.move_worker(worker, WAREHOUSE, occupancy)
                continue

            ship = None
            if self.agent == "utility":
                # Re-score every tick; lowest ship ID breaks score ties.
                ship = max(
                    available_ships,
                    key=lambda item: (self.berth_utility(worker, item),
                                      -item["id"]),
                )
                # Safety guard: if a risky berth wins, wait rather than
                # serve it. Carrying workers still return by safe routes.
                if (worker["avoid_unsafe"]
                        and ship["berth"] in ("B1", "B4")):
                    worker["target_berth"] = None
                    worker["target_ship_id"] = None
                    self.move_worker(worker, WAREHOUSE, occupancy)
                    continue

            if self.agent == "model":
                # Keep the previous assignment while the remembered ship
                # remains at that berth and the berth is eligible.
                remembered = worker["berth_memory"].get(
                    worker["target_berth"], {}
                )
                if remembered.get("ship_id") == worker["target_ship_id"]:
                    ship = next(
                        (candidate for candidate in available_ships
                         if candidate["berth"] == worker["target_berth"]
                         and candidate["id"] == worker["target_ship_id"]),
                        None,
                    )

            if ship is None:
                ship = min(
                    available_ships,
                    key=lambda item: (
                        distance(worker["pos"], BERTHS[item["berth"]]),
                        item["id"],
                    ),
                )

            if self.agent in ("model", "utility"):
                worker["target_berth"] = ship["berth"]
                worker["target_ship_id"] = ship["id"]

            berth_position = BERTHS[ship["berth"]]
            if distance(worker["pos"], berth_position) == 1:
                ship["left"] -= 1
                worker["carrying"] = True
                if ship["left"] == 0:
                    self.log(tick, f"Ship {ship['id']} is empty and leaves.")
                continue

            self.move_worker(worker, berth_position, occupancy)

    def measure_crowding(self):
        occupancy = Counter(worker["pos"] for worker in self.workers)
        self.crowding_violations += sum(
            count > 3 for count in occupancy.values()
        )

    def run(self):
        for tick in range(TOTAL_TICKS):
            self.update_weather(tick)
            self.update_ships(tick)
            self.update_workers(tick)
            self.measure_crowding()

        # Include elapsed waiting for ships still undocked at tick 600.
        # These are censored observations, not their final waiting times.
        waits = [
            (ship["dock"] if ship["dock"] is not None else TOTAL_TICKS)
            - ship["arrival"]
            for ship in self.ships
        ]
        average_wait = sum(waits) / len(waits) if waits else 0.0
        return {
            "cargo_delivered": self.cargo_delivered,
            "avg_ship_wait": round(average_wait, 3),
            "crowding_violations": self.crowding_violations,
            "storm_knockouts": self.storm_knockouts,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, choices=range(10), default=3,
                        help="Last Student ID digit; default 3 for 231743.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--agent", choices=["baseline", "reflex", "model", "utility", "llm"],
                        default="reflex")
    parser.add_argument("--verbose", action="store_true")
    # These options apply only to the LLM Harbour Master.
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-file", default="logs/llm_log.jsonl")
    parser.add_argument("--replay", help="Replay a saved LLM JSONL log; no new calls.")
    parser.add_argument("--replay-run-id")
    parser.add_argument("--approve", action="store_true",
                        help="Ask for protected orders in the terminal; otherwise block them.")
    parser.add_argument("--max-ticks", type=int, default=600,
                        help="LLM only: use 40 for a smoke test, 600 for evaluation.")
    parser.add_argument("--actions-per-turn", type=int, default=1,
                        help="LLM only: 1 to 3 sequential actions at each 20-tick turn.")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--untrusted-file")
    args = parser.parse_args()
    if args.agent == "llm":
        from llm_master import run_llm
        try:
            metrics = run_llm(args)
        except (RuntimeError, ValueError, OSError) as error:
            parser.exit(1, str(error) + "\n")
        print(json.dumps(metrics))
        return
    harbour = Harbour(args.digit, args.seed, args.agent, args.verbose)
    print(json.dumps(harbour.run()))


if __name__ == "__main__":
    main()
