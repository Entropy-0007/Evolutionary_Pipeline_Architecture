#!/usr/bin/env python3
"""
run_pipeline_evo.py
Single-file prototype: simulator + GA + simple ACO + plotting.

Usage:
    python run_pipeline_evo.py
Outputs:
    - fitness_history.png
    - metrics_comparison.png
    - pipeline_best.png
    - ga_history.csv
"""

import random
import math
import csv
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Configuration (fast defaults)
# ---------------------------
RANDOM_SEED = 42
TRACE_LEN = 2000              # small for speed
POP_SIZE = 32
GENERATIONS = 20
ELITISM = 2
TOURNAMENT = 3
MUTATION_RATE = 0.08
ACO_ANTS = 8
ACO_ITERS = 8

OUTPUT_DIR = "evo_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------
# Workload generator
# ---------------------------
def generate_synthetic_trace(length=TRACE_LEN, mix=(0.6, 0.25, 0.15)):
    """
    Returns list of instructions: each instruction is a dict with:
    - type: 'ALU' or 'LOAD' or 'BR'
    - dst: destination register id or None for branch
    - srcs: list of source registers
    """
    trace = []
    regs = 32
    alu_p, load_p, br_p = mix
    for i in range(length):
        r = random.random()
        if r < alu_p:
            itype = 'ALU'
            lat = 1
        elif r < alu_p + load_p:
            itype = 'LOAD'
            lat = 4  # memory latency proxy
        else:
            itype = 'BR'
            lat = 1
        # pick registers (simple)
        dst = random.randint(0, regs-1) if itype != 'BR' else None
        srcs = [random.randint(0, regs-1) for _ in range(2)]
        trace.append({'type': itype, 'dst': dst, 'srcs': srcs, 'lat': lat})
    return trace


# ---------------------------
# Genome and utilities
# ---------------------------
@dataclass
class Genome:
    stage_count: int               # 3..8
    per_stage_latency: List[int]   # len == stage_count, each 1..4
    width: int                     # 1..4
    forwarding: int                # 0 or 1
    rob_class: int                 # 0..3
    # runtime fields
    fitness: float = field(default=0.0)
    metrics: Dict = field(default_factory=dict)

def random_genome():
    sc = random.randint(3, 6)
    lat = [random.randint(1, 4) for _ in range(sc)]
    w = random.randint(1, 3)
    f = random.randint(0, 1)
    rob = random.randint(0, 3)
    return Genome(stage_count=sc, per_stage_latency=lat, width=w, forwarding=f, rob_class=rob)

def genome_to_str(g: Genome):
    return f"S{g.stage_count}-W{g.width}-F{g.forwarding}-ROB{g.rob_class}-L{''.join(map(str,g.per_stage_latency))}"


# ---------------------------
# Simple cycle-based simulator
# ---------------------------
def simulate(genome: Genome, trace: List[Dict], forwarding_topology: np.ndarray = None, max_cycles=200000):
    stages = genome.stage_count
    per_stage_latency = genome.per_stage_latency
    width = genome.width
    forwarding_enabled = bool(genome.forwarding)

    if forwarding_topology is None:
        forwarding_topology = np.ones((stages, stages), dtype=int) if forwarding_enabled else np.zeros((stages, stages), dtype=int)

    regs_ready = [0]*128
    reg_producer_stage = [None]*128  # NEW: track producer stage for each register
    pc = 0
    cycles = 0
    committed = 0
    in_flight = []
    stall_cycles = 0
    idle_cycles = 0
    max_instr = len(trace)
    commit_stage = stages - 1

    while committed < max_instr and cycles < max_cycles:
        cycles += 1
        committed_this_cycle = 0

        # Fetch stage
        for _ in range(width):
            if pc >= max_instr:
                break
            count_stage0 = sum(1 for ins in in_flight if ins['stage_idx'] == 0)
            if count_stage0 >= width:
                break
            inst = trace[pc]
            hazard = False
            for src in inst['srcs']:
                if regs_ready[src] > cycles:
                    producer_stage = reg_producer_stage[src]
                    if not forwarding_enabled or producer_stage is None or forwarding_topology[producer_stage][0] == 0:
                        hazard = True
                        break
            if hazard:
                stall_cycles += 1
                break
            in_flight.append({'it': inst.copy(), 'stage_idx': 0, 'remaining': per_stage_latency[0], 'fetch_cycle': cycles})
            if inst['dst'] is not None:
                reg_producer_stage[inst['dst']] = 0
            pc += 1

        # Advance instructions
        for ins in sorted(in_flight, key=lambda x: -x['stage_idx']):
            if ins['remaining'] > 1:
                ins['remaining'] -= 1
                continue
            stage_idx = ins['stage_idx']
            inst = ins['it']
            if stage_idx == commit_stage:
                committed += 1
                committed_this_cycle += 1
                if inst['dst'] is not None:
                    regs_ready[inst['dst']] = cycles + 1
                    reg_producer_stage[inst['dst']] = None
                in_flight.remove(ins)
                continue
            next_idx = stage_idx + 1
            count_next = sum(1 for x in in_flight if x['stage_idx'] == next_idx)
            if count_next >= width:
                stall_cycles += 1
                continue
            hazard = False
            for src in inst['srcs']:
                if regs_ready[src] > cycles:
                    producer_stage = reg_producer_stage[src]
                    if not forwarding_enabled or producer_stage is None or forwarding_topology[producer_stage][next_idx] == 0:
                        hazard = True
                        break
            if hazard:
                stall_cycles += 1
                continue
            ins['stage_idx'] = next_idx
            ins['remaining'] = per_stage_latency[next_idx]
            if inst['dst'] is not None:
                reg_producer_stage[inst['dst']] = next_idx

        if len(in_flight) == 0 and pc >= max_instr:
            break
        if committed_this_cycle == 0 and len(in_flight) == 0 and pc < max_instr:
            idle_cycles += 1

    ipc = committed / cycles if cycles > 0 else 0.0
    throughput = committed / cycles if cycles > 0 else 0.0
    metrics = {
        'cycles': cycles,
        'committed': committed,
        'stalls': stall_cycles,
        'idle_cycles': idle_cycles,
        'IPC': ipc,
        'throughput': throughput
    }
    return metrics


# ---------------------------
# Fitness and cost model
# ---------------------------
def hardware_cost(genome: Genome):
    # simple proxy: number of stages * width + ROB class penalty
    return genome.stage_count * genome.width + (genome.rob_class + 1) * 2

def evaluate_genome(genome: Genome, trace: List[Dict], aco_topology: np.ndarray = None):
    metrics = simulate(genome, trace, forwarding_topology=aco_topology)
    ipc = metrics['IPC']
    cost = hardware_cost(genome)
    # fitness: maximize IPC, penalize cost and stalls
    fitness = ipc * 100.0 - 0.5 * cost - 0.01 * metrics['stalls']
    genome.fitness = fitness
    genome.metrics = metrics
    return fitness, metrics


# ---------------------------
# GA operators
# ---------------------------
def tournament_select(pop: List[Genome], k=TOURNAMENT):
    best = None
    for _ in range(k):
        cand = random.choice(pop)
        if best is None or cand.fitness > best.fitness:
            best = cand
    return best

def crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    # one-point crossover on flattened representation
    # flatten: [stage_count, per_stage_latency..., width, forwarding, rob_class]
    la = [a.stage_count] + a.per_stage_latency + [a.width, a.forwarding, a.rob_class]
    lb = [b.stage_count] + b.per_stage_latency + [b.width, b.forwarding, b.rob_class]
    # pad shorter with zeros (safe in our small setting)
    L = min(len(la), len(lb))
    if L <= 3:
        return random_genome(), random_genome()
    pt = random.randint(1, L-2)
    ca = la[:pt] + lb[pt:]
    cb = lb[:pt] + la[pt:]
    # reconstruct genomes heuristically
    def from_flat(f):
        sc = max(3, min(6, int(round(f[0]))))
        # next sc entries as latencies (if missing, fill with 1)
        lat = []
        idx = 1
        for i in range(sc):
            if idx < len(f):
                lat.append(max(1, min(4, int(round(f[idx])))))
            else:
                lat.append(1)
            idx += 1
        width = max(1, min(3, int(round(f[idx])) if idx < len(f) else 1)); idx += 1
        forwarding = int(round(f[idx])) if idx < len(f) else 0; idx += 1
        rob = int(round(f[idx])) if idx < len(f) else 0
        return Genome(stage_count=sc, per_stage_latency=lat, width=width, forwarding=forwarding, rob_class=max(0,min(3,rob)))
    return from_flat(ca), from_flat(cb)

def mutate(g: Genome, rate=MUTATION_RATE):
    if random.random() < rate:
        # mutate stage_count
        g.stage_count = max(3, min(6, g.stage_count + random.choice([-1, 1])))
        # adjust per_stage_latency length
        while len(g.per_stage_latency) < g.stage_count:
            g.per_stage_latency.append(random.randint(1, 4))
        while len(g.per_stage_latency) > g.stage_count:
            g.per_stage_latency.pop()
    # mutate latencies
    for i in range(len(g.per_stage_latency)):
        if random.random() < rate:
            g.per_stage_latency[i] = max(1, min(4, g.per_stage_latency[i] + random.choice([-1, 1])))
    if random.random() < rate:
        g.width = max(1, min(3, g.width + random.choice([-1, 1])))
    if random.random() < rate:
        g.forwarding = 1 - g.forwarding
    if random.random() < rate:
        g.rob_class = max(0, min(3, g.rob_class + random.choice([-1, 1])))
    return g

# ---------------------------
# Simple ACO (forwarding topology search)
# ---------------------------
def aco_optimize_forwarding(genome: Genome, trace: List[Dict], ants=ACO_ANTS, iters=ACO_ITERS):
    stages = genome.stage_count
    # pheromone matrix for pairwise forwarding (stage_from, stage_to)
    pher = np.ones((stages, stages), dtype=float)
    best_top = np.zeros((stages, stages), dtype=int)
    best_score = -1e9
    for it in range(iters):
        ant_solutions = []
        for a in range(ants):
            # probabilistically decide forwarding edges based on pheromone
            prob = pher / pher.sum(axis=1, keepdims=True)
            # construct adjacency by sampling per row
            top = np.zeros((stages, stages), dtype=int)
            for i in range(stages):
                # each ant chooses k outgoing edges (k in [0..2])
                k = random.choices([0,1,2], weights=[0.3,0.5,0.2])[0]
                if k == 0:
                    continue
                choices = np.random.choice(stages, size=k, replace=False, p=prob[i])
                for c in choices:
                    top[i, c] = 1
            # ensure diagonal (forward from stage to itself) is allowed
            for s in range(stages):
                top[s, s] = 1
            ant_solutions.append(top)
        # evaluate ants
        scores = []
        for top in ant_solutions:
            fitness, metrics = evaluate_genome(genome, trace, aco_topology=top)
            score = fitness
            scores.append(score)
            if score > best_score:
                best_score = score
                best_top = top.copy()
        # pheromone update: evaporate then reinforce top solutions
        pher *= 0.9
        # reinforce proportional to score (shift to positive)
        min_s = min(scores)
        adj_scores = [s - min_s + 1e-3 for s in scores]
        for top, sc in zip(ant_solutions, adj_scores):
            pher += top * (sc / sum(adj_scores))
    # return best topology found and its evaluated metrics
    best_fit, best_metrics = evaluate_genome(genome, trace, aco_topology=best_top)
    return best_top, best_metrics

# ---------------------------
# GA main loop
# ---------------------------
def ga_evolve(trace: List[Dict], pop_size=POP_SIZE, generations=GENERATIONS):
    # init pop
    pop = [random_genome() for _ in range(pop_size)]
    # evaluate initial
    for g in pop:
        evaluate_genome(g, trace)
    history = []
    for gen in range(generations):
        pop.sort(key=lambda x: -x.fitness)
        best = pop[0]
        mean_fit = sum(p.fitness for p in pop) / len(pop)
        history.append({'gen': gen, 'best_fit': best.fitness, 'mean_fit': mean_fit, 'best_str': genome_to_str(best), 'best_metrics': best.metrics})
        # print progress
        print(f"[GA] Gen {gen}: best={best.fitness:.3f} IPC={best.metrics.get('IPC',0):.3f} stalls={best.metrics.get('stalls',0)} best={genome_to_str(best)}")
        # create new population
        newpop = pop[:ELITISM]  # elitism
        while len(newpop) < pop_size:
            p1 = tournament_select(pop)
            p2 = tournament_select(pop)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            evaluate_genome(c1, trace)
            evaluate_genome(c2, trace)
            newpop.extend([c1, c2])
        pop = newpop[:pop_size]
    pop.sort(key=lambda x: -x.fitness)
    return pop, history

# ---------------------------
# Utilities: plotting and csv export
# ---------------------------
def save_history_csv(history, filename):
    keys = ['gen', 'best_fit', 'mean_fit', 'best_str', 'best_metrics']
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gen', 'best_fit', 'mean_fit', 'best_str', 'IPC', 'stalls', 'cycles', 'committed'])
        for h in history:
            bm = h['best_metrics']
            w.writerow([h['gen'], h['best_fit'], h['mean_fit'], h['best_str'], bm.get('IPC',0), bm.get('stalls',0), bm.get('cycles',0), bm.get('committed',0)])

def plot_history(history, outpath):
    gens = [h['gen'] for h in history]
    best = [h['best_fit'] for h in history]
    mean = [h['mean_fit'] for h in history]
    plt.figure(figsize=(8,4))
    plt.plot(gens, best, label='best fitness')
    plt.plot(gens, mean, label='mean fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('GA Fitness over Generations')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_metrics_comparison(baseline_metrics, ga_metrics, gaaco_metrics, outpath):
    labels = ['Baseline', 'GA-only', 'GA+ACO']
    ipc = [baseline_metrics['IPC'], ga_metrics['IPC'], gaaco_metrics['IPC']]
    stalls = [baseline_metrics['stalls'], ga_metrics['stalls'], gaaco_metrics['stalls']]
    idle = [baseline_metrics['idle_cycles'], ga_metrics['idle_cycles'], gaaco_metrics['idle_cycles']]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8,4))
    plt.bar(x - width, ipc, width, label='IPC')
    plt.bar(x, [s / 1000.0 for s in stalls], width, label='stalls(k/1000)')
    plt.bar(x + width, idle, width, label='idle cycles')
    plt.xticks(x, labels)
    plt.ylabel('Metric value (IPC or scaled)')
    plt.legend()
    plt.title('Metrics Comparison')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def draw_pipeline(genome: Genome, outpath):
    fig, ax = plt.subplots(figsize=(8,2))
    ax.axis('off')
    n = genome.stage_count
    for i in range(n):
        rect = plt.Rectangle((i*1.2, 0), 1.0, 0.8, fill=True, color='#66c2a5', ec='k')
        ax.add_patch(rect)
        ax.text(i*1.2 + 0.5, 0.4, f"S{i}\nlat={genome.per_stage_latency[i]}", ha='center', va='center')
    ax.set_xlim(-0.5, n*1.2)
    ax.set_ylim(-0.5, 1.5)
    plt.title("Best Pipeline Schematic")
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------------------------
# Main orchestration
# ---------------------------
def main():
    print("Generating synthetic trace...")
    trace = generate_synthetic_trace()
    print("Running baseline fixed 5-stage pipeline...")
    baseline_genome = Genome(stage_count=5, per_stage_latency=[1,1,1,1,1], width=1, forwarding=0, rob_class=1)
    baseline_metrics = simulate(baseline_genome, trace)
    print(f"Baseline metrics: IPC={baseline_metrics['IPC']:.3f}, stalls={baseline_metrics['stalls']}, cycles={baseline_metrics['cycles']}")

    print("\nRunning GA-only evolution (no ACO local search)...")
    pop, history = ga_evolve(trace)
    save_history_csv(history, os.path.join(OUTPUT_DIR, 'ga_history.csv'))
    plot_history(history, os.path.join(OUTPUT_DIR, 'fitness_history.png'))
    best_ga = pop[0]
    ga_metrics = best_ga.metrics
    print(f"GA best: {genome_to_str(best_ga)} IPC={ga_metrics['IPC']:.3f} stalls={ga_metrics['stalls']}")

    print("\nRunning GA + ACO (local forwarding optimization on GA best)...")
    best_top, best_aco_metrics = aco_optimize_forwarding(best_ga, trace)
    print(f"GA+ACO metrics: IPC={best_aco_metrics['IPC']:.3f}, stalls={best_aco_metrics['stalls']}")
    # Prepare comparison plots
    plot_metrics_comparison(baseline_metrics, ga_metrics, best_aco_metrics, os.path.join(OUTPUT_DIR, 'metrics_comparison.png'))
    draw_pipeline(best_ga, os.path.join(OUTPUT_DIR, 'pipeline_best.png'))

    # Save quick summary CSV
    with open(os.path.join(OUTPUT_DIR, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'IPC', 'stalls', 'idle_cycles', 'cycles', 'committed'])
        w.writerow(['baseline', baseline_metrics['IPC'], baseline_metrics['stalls'], baseline_metrics['idle_cycles'], baseline_metrics['cycles'], baseline_metrics['committed']])
        w.writerow(['ga_best', ga_metrics['IPC'], ga_metrics['stalls'], ga_metrics['idle_cycles'], ga_metrics['cycles'], ga_metrics['committed']])
        w.writerow(['ga_aco', best_aco_metrics['IPC'], best_aco_metrics['stalls'], best_aco_metrics['idle_cycles'], best_aco_metrics['cycles'], best_aco_metrics['committed']])
    print(f"\nOutputs saved to {OUTPUT_DIR}: fitness_history.png, metrics_comparison.png, pipeline_best.png, ga_history.csv, summary.csv")
    print("Done.")

if __name__ == '__main__':
    main()
