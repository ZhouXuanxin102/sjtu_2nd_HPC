from __future__ import annotations

import os
import sys
from collections import defaultdict

from rune_scheduler.api.v1 import (
    DispatchAction,
    DispatchDecision,
    Observation,
    Placement,
    PublicScenarioModel,
    TaskType,
    TruncatedGeometricFailurePrior,
)


def _mid(bound: object) -> float:
    return (bound.min + bound.max) / 2


class Policy:
    """在线调度：按关键路径选类型，在合法 Worker 上尽量同批共享 setup。"""

    def __init__(self, model: PublicScenarioModel) -> None:
        self._max_batch = model.safety_limits.max_batch_size
        self._types = {item.id: item for item in model.task_types}
        self._workers = {item.id: item for item in model.workers}
        self._mult: dict[tuple[str, str], float] = {}
        for worker in model.workers:
            for item in worker.duration_multipliers:
                self._mult[(worker.id, item.task_type)] = (
                    item.ratio.numerator / item.ratio.denominator
                )
        self._eligible: dict[str, tuple[str, ...]] = {}
        for task_type in model.task_types:
            need = set(task_type.required_capabilities)
            self._eligible[task_type.id] = tuple(
                worker.id
                for worker in model.workers
                if need <= set(worker.capabilities)
            )
        self._runtime = {
            task_type.id: _mid(task_type.runtime_work) for task_type in model.task_types
        }
        self._fail = {
            task_type.id: _failure_factors(task_type) for task_type in model.task_types
        }
        self._duty: dict[str, float] = {}
        self._uptime: dict[str, float] = {}
        for worker in model.workers:
            prior = worker.outage_prior
            if prior is None:
                self._duty[worker.id] = 0.0
                self._uptime[worker.id] = 1e18
                continue
            duration = _mid(prior.duration)
            uptime = _mid(prior.uptime)
            self._duty[worker.id] = (
                duration / (duration + uptime) if duration + uptime else 0.0
            )
            self._uptime[worker.id] = uptime
        self._finish: dict[tuple[str, str], float] = {}
        self._pool_finish: dict[tuple[str, str], float] = {}
        self._pool_anchor: dict[str, float] = {}
        self._unit: dict[str, float] = {}
        for task_type in model.task_types:
            best = None
            best_worker = None
            for worker_id in self._eligible[task_type.id]:
                estimate = self._estimate(worker_id, task_type)
                self._finish[(worker_id, task_type.id)] = estimate
                if best is None or estimate < best:
                    best = estimate
                    best_worker = worker_id
            self._unit[task_type.id] = best if best is not None else 1e9
            anchor = (
                self._pack_cost(best_worker, task_type)
                if best_worker is not None
                else 1e9
            )
            for worker_id in self._eligible[task_type.id]:
                self._pool_finish[(worker_id, task_type.id)] = self._pack_cost(
                    worker_id, task_type
                )
            self._pool_anchor[task_type.id] = anchor
        self._best_finish = {
            task_type: min(
                self._finish[(worker_id, task_type)]
                for worker_id in workers
                if (worker_id, task_type) in self._finish
            )
            if any((worker_id, task_type) in self._finish for worker_id in workers)
            else 1e9
            for task_type, workers in self._eligible.items()
        }
        self._rank = self._build_rank(model)
        self._node_bias = self._build_node_bias(model)
        soon: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
        seen: set[tuple[str, str, str]] = set()

        def note(parent: str, successor: str) -> None:
            owners = self._eligible.get(successor, ())
            if not owners:
                return
            ranked = sorted(
                owners,
                key=lambda worker_id: (self._finish[(worker_id, successor)], worker_id),
            )
            best = self._finish[(ranked[0], successor)]
            second = self._finish[(ranked[1], successor)] if len(ranked) > 1 else 1e9
            if second <= best * 1.4:
                return
            key = (ranked[0], parent, successor)
            if key in seen:
                return
            seen.add(key)
            soon[ranked[0]].append((parent, successor, len(owners) == 1))

        for derivation in model.derivations:
            note(derivation.parent_type, derivation.successor_type)
        for template in model.workflow_templates:
            nodes = {node.id: node for node in template.nodes}
            for node in template.nodes:
                for parent_id in node.depends_on:
                    note(nodes[parent_id].task_type, node.task_type)
        self._soon = {key: tuple(value) for key, value in soon.items()}
        self._guarded: dict[str, frozenset[str]] = defaultdict(frozenset)
        guarded: dict[str, set[str]] = defaultdict(set)
        for task_type in model.task_types:
            if not task_type.required_capabilities:
                continue
            owners = self._eligible[task_type.id]
            if not (0 < len(owners) <= 2):
                continue
            best = self._best_finish[task_type.id]
            for worker_id in owners:
                finish = self._finish.get((worker_id, task_type.id), 1e9)
                if finish < best * 2.0:
                    guarded[worker_id].add(task_type.id)
        self._guarded = {key: frozenset(value) for key, value in guarded.items()}
        self._release_home: dict[str, frozenset[str]] = {}
        for task_type in model.task_types:
            owners = self._eligible[task_type.id]
            if not (task_type.required_capabilities and 0 < len(owners) <= 2):
                continue
            best = self._best_finish[task_type.id]
            fast = tuple(
                worker_id
                for worker_id in owners
                if self._finish.get((worker_id, task_type.id), 1e9) < best * 2.0
            )
            if fast:
                self._release_home[task_type.id] = frozenset(fast)
        self._releases = tuple(
            (_mid(item.at), item.task_type) for item in model.releases
        )
        self._calls = 0
        self._last_time: int | None = None

    def choose_placements(self, observation: Observation) -> DispatchDecision:
        load = {
            worker.worker_id: [worker.cpu_demand, worker.memory_reserved, worker.available]
            for worker in observation.workers
        }
        self._hot = {task.task_type for task in observation.ready_tasks}
        self._hot.update(attempt.task_type for attempt in observation.active_attempts)
        grouped: dict[str, list] = defaultdict(list)
        for task in observation.ready_tasks:
            grouped[task.task_type].append(task)
        for tasks in grouped.values():
            tasks.sort(key=self._task_key)
        used: set[str] = set()
        placements: list[Placement] = []
        slots = self._max_batch
        while slots > 0:
            cohort = self._next_cohort(
                grouped, load, used, slots, observation.logical_time
            )
            if not cohort:
                break
            placements.extend(cohort)
            slots -= len(cohort)
        self._trace(observation, len(placements))
        if not placements:
            return DispatchDecision(DispatchAction.DEFER, ())
        return DispatchDecision(DispatchAction.DISPATCH, tuple(placements))

    def _next_cohort(
        self,
        grouped: dict[str, list],
        load: dict[str, list],
        used: set[str],
        slots: int,
        now: int,
    ) -> list[Placement]:
        ready_types = {
            kind
            for kind, tasks in grouped.items()
            if any(task.task_id not in used for task in tasks)
        }
        order = sorted(ready_types, key=lambda task_type: (-self._rank[task_type], task_type))
        for task_type in order:
            cohort = self._place_type(
                task_type, grouped[task_type], load, used, slots, now, ready_types
            )
            if cohort:
                return cohort
        return []

    def _place_type(
        self,
        task_type: str,
        tasks: list,
        load: dict[str, list],
        used: set[str],
        slots: int,
        now: int,
        ready_types: set[str],
    ) -> list[Placement]:
        pending = [task for task in tasks if task.task_id not in used]
        if not pending:
            return []
        chosen = self._best_assignment(
            task_type, len(pending), load, slots, now, ready_types, True
        )
        if chosen is None:
            chosen = self._best_assignment(
                task_type, len(pending), load, slots, now, ready_types, False
            )
        if chosen is None:
            return []
        worker_id, count = chosen
        spec = self._types[task_type]
        cpu, memory, available = load[worker_id]
        load[worker_id] = [
            cpu + count * spec.cpu_demand,
            memory + count * spec.memory_reservation,
            available,
        ]
        for task in pending[:count]:
            used.add(task.task_id)
        return [Placement(task.task_id, worker_id) for task in pending[:count]]

    def _best_assignment(
        self,
        task_type: str,
        pending: int,
        load: dict[str, list],
        slots: int,
        now: int,
        ready_types: set[str],
        honor_reserve: bool,
    ) -> tuple[str, int] | None:
        spec = self._types[task_type]
        best_key: tuple[float, float, str] | None = None
        best: tuple[str, int] | None = None
        for worker_id in self._speed_pool(task_type, load, pending, slots):
            state = load.get(worker_id)
            if state is None or not state[2]:
                continue
            count = _fits(spec, self._workers[worker_id], state[0], state[1], pending, slots)
            if count < 1:
                continue
            # One free slot starts a setup group of one. If the idle worker
            # could hold several, wait for that pack instead of paying setup now.
            if count == 1 and pending >= 3 and spec.setup_work >= self._runtime[task_type]:
                empty = _fits(spec, self._workers[worker_id], 0, 0, pending, slots)
                if empty >= 3:
                    continue
            finish = self._finish[(worker_id, task_type)]
            if self._soon_blocked(worker_id, task_type, ready_types) and self._defer_safe(
                load, now
            ):
                continue
            if honor_reserve and self._reserved(
                worker_id, task_type, now, finish, ready_types
            ):
                continue
            key = (-(count / finish), finish, worker_id)
            if best_key is None or key < best_key:
                best_key = key
                best = (worker_id, count)
        return best

    def _speed_pool(
        self, task_type: str, load: dict[str, list], pending: int, slots: int
    ) -> tuple[str, ...]:
        """Keep tasks on near-best workers while one of them has room.

        A worker just outside the fast pool is used only when every fast
        worker is up but full. That covers a 2x worker whose outage adjustment
        pushes it slightly past 2.5, without opening the 3x tier.
        """
        eligible = self._eligible[task_type]
        best = self._pool_anchor.get(task_type, 1e9)

        def band(limit: float) -> tuple[str, ...]:
            return tuple(
                worker_id
                for worker_id in eligible
                if self._pool_finish.get((worker_id, task_type), 1e9) <= best * limit
            )

        fast = band(2.5)
        if not fast:
            return eligible
        if self._has_room(task_type, fast, load, pending, slots):
            return fast
        if any(load.get(worker_id, (0, 0, False))[2] for worker_id in fast):
            if pending >= 3:
                wider = band(2.6)
                if self._has_room(task_type, wider, load, pending, slots):
                    return wider
            return fast
        return eligible

    def _has_room(
        self,
        task_type: str,
        workers: tuple[str, ...],
        load: dict[str, list],
        pending: int,
        slots: int,
    ) -> bool:
        spec = self._types[task_type]
        for worker_id in workers:
            state = load.get(worker_id)
            if state is None or not state[2]:
                continue
            if _fits(spec, self._workers[worker_id], state[0], state[1], pending, slots) >= 1:
                return True
        return False

    def _defer_safe(self, load: dict[str, list], now: int) -> bool:
        if any(now < at for at, _release in self._releases):
            return True
        return any(not state[2] or state[0] or state[1] for state in load.values())

    def _soon_blocked(self, worker_id: str, task_type: str, ready_types: set[str]) -> bool:
        if self._finish.get((worker_id, task_type), 1e9) <= self._best_finish.get(task_type, 0.0) * 1.05:
            return False
        for parent, successor, sole in self._soon.get(worker_id, ()):
            if task_type == successor:
                continue
            if successor in ready_types or (sole and parent in self._hot):
                return True
            if self._duty.get(worker_id, 0.0) > 0 and (
                successor in self._hot or parent in self._hot
            ):
                return True
        return False

    def _reserved(
        self,
        worker_id: str,
        task_type: str,
        now: int,
        finish: float,
        ready_types: set[str],
    ) -> bool:
        guarded = self._guarded.get(worker_id, frozenset())
        if task_type not in guarded and any(kind in ready_types for kind in guarded):
            return True
        for at, release_type in self._releases:
            if release_type == task_type or now >= at:
                continue
            homes = self._release_home.get(release_type)
            if not homes or worker_id not in homes:
                continue
            if self._rank[release_type] <= self._rank[task_type]:
                continue
            if now + finish > at:
                return True
        return False

    def _pack_cost(self, worker_id: str, task_type: TaskType) -> float:
        """Outage-adjusted cost used only to widen the pool of setup-heavy types.

        Ranking still uses the milder finish, so a fast worker stays preferred
        whenever it can take the task. A stable worker joins the pool when a
        wiped attempt on the risky worker is no longer clearly shorter.
        """
        if task_type.setup_work < self._runtime[task_type.id]:
            return self._finish[(worker_id, task_type.id)]
        duty = self._duty[worker_id]
        base = self._work(worker_id, task_type)
        if duty <= 0 or self._uptime[worker_id] >= 1e17:
            return base
        return base / max(1e-3, 1.0 - duty)

    def _work(self, worker_id: str, task_type: TaskType) -> float:
        setup_factor, runtime_factor = self._fail[task_type.id]
        work = task_type.setup_work * setup_factor + self._runtime[task_type.id] * runtime_factor
        multiplier = self._mult.get((worker_id, task_type.id), 1.0)
        worker = self._workers[worker_id]
        factor = 1.0
        if task_type.cpu_demand > worker.cpu_capacity > 0:
            factor = worker.cpu_capacity / task_type.cpu_demand
        return max(work * multiplier / factor, 1e-6)

    def _estimate(self, worker_id: str, task_type: TaskType) -> float:
        setup_factor, runtime_factor = self._fail[task_type.id]
        work = task_type.setup_work * setup_factor + self._runtime[task_type.id] * runtime_factor
        multiplier = self._mult.get((worker_id, task_type.id), 1.0)
        worker = self._workers[worker_id]
        factor = 1.0
        if task_type.cpu_demand > worker.cpu_capacity > 0:
            factor = worker.cpu_capacity / task_type.cpu_demand
        estimate = work * multiplier / factor
        duty = self._duty[worker_id]
        uptime = self._uptime[worker_id]
        if duty > 0 and uptime < 1e17:
            estimate *= 1.0 + min(0.9, duty * estimate / uptime)
        return max(estimate, 1e-6)

    def _build_rank(self, model: PublicScenarioModel) -> dict[str, float]:
        edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for template in model.workflow_templates:
            nodes = {node.id: node for node in template.nodes}
            for node in template.nodes:
                for parent in node.depends_on:
                    edges[nodes[parent].task_type][node.task_type] += 1.0
        for derivation in model.derivations:
            edges[derivation.parent_type][derivation.successor_type] += _mid(
                derivation.fanout
            )
        rank: dict[str, float] = {}
        visiting: set[str] = set()

        def visit(task_type: str) -> float:
            if task_type in rank:
                return rank[task_type]
            if task_type in visiting:
                return self._unit.get(task_type, 0.0)
            visiting.add(task_type)
            total = self._unit.get(task_type, 0.0)
            for successor, weight in edges.get(task_type, {}).items():
                total += weight * visit(successor)
            visiting.remove(task_type)
            rank[task_type] = total
            return total

        for task_type in self._types:
            visit(task_type)
        return rank

    def _build_node_bias(self, model: PublicScenarioModel) -> dict[str, float]:
        bias: dict[str, float] = {}
        for template in model.workflow_templates:
            nodes = {node.id: node for node in template.nodes}
            children: dict[str, list[str]] = defaultdict(list)
            for node in template.nodes:
                for parent in node.depends_on:
                    children[parent].append(node.id)
            memo: dict[str, float] = {}

            def visit(node_id: str) -> float:
                if node_id in memo:
                    return memo[node_id]
                memo[node_id] = 0.0
                node = nodes[node_id]
                total = self._unit.get(node.task_type, 0.0)
                for child in children[node_id]:
                    total += visit(child)
                memo[node_id] = total
                return total

            for node_id in nodes:
                value = visit(node_id)
                if value > bias.get(node_id, 0.0):
                    bias[node_id] = value
        return bias

    def _task_key(self, task: object) -> tuple:
        return (
            -task.failed_attempts,
            -task.interrupted_attempts,
            -self._node_bias.get(task.workflow_node_id or "", 0.0),
            task.task_id,
        )

    def _trace(self, observation: Observation, placed: int) -> None:
        self._calls += 1
        trace = os.environ.get("RUNE_TRACE")
        if trace not in {"1", "all"} or (trace == "1" and self._calls > 8):
            self._last_time = observation.logical_time
            return
        same = self._last_time is not None and observation.logical_time == self._last_time
        print(
            f"t={observation.logical_time} d={observation.decision_id} "
            f"same={int(bool(same))} ready={len(observation.ready_tasks)} placed={placed}",
            file=sys.stderr,
            flush=True,
        )
        self._last_time = observation.logical_time


def _failure_factors(task_type: TaskType) -> tuple[float, float]:
    prior = task_type.failure_prior
    if not isinstance(prior, TruncatedGeometricFailurePrior):
        return 1.0, 1.0
    probability = prior.failure_probability.numerator / prior.failure_probability.denominator
    limit = prior.max_failures
    if probability <= 0 or limit <= 0:
        return 1.0, 1.0
    if probability >= 1:
        expected_failures = float(limit)
    else:
        expected_failures = probability * (1 - probability**limit) / (1 - probability)
    fraction = (
        prior.failure_fraction.min.numerator / prior.failure_fraction.min.denominator
        + prior.failure_fraction.max.numerator / prior.failure_fraction.max.denominator
    ) / 2
    return 1.0 + expected_failures, 1.0 + expected_failures * fraction


def _fits(
    task_type: TaskType,
    worker: object,
    cpu_used: int,
    memory_used: int,
    pending: int,
    slots: int,
) -> int:
    memory_need = task_type.memory_reservation
    if memory_need <= 0:
        by_memory = pending
    else:
        by_memory = (worker.memory_capacity - memory_used) // memory_need
    if by_memory < 1:
        return 0
    cpu_need = task_type.cpu_demand
    if cpu_need <= 0:
        by_cpu = pending
    else:
        by_cpu = (worker.cpu_capacity - cpu_used) // cpu_need
    if by_cpu >= 1:
        return min(pending, slots, by_memory, by_cpu)
    if cpu_used == 0 and worker.cpu_capacity > 0 and min(pending, slots, by_memory) >= 1:
        return 1
    return 0
