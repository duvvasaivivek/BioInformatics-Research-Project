"""
Novelty 2 Inference CLI: Two-Stage Sequential Cascaded Sieve.

Evaluates molecules hierarchically:
  Stage 1: High-Throughput BBB Permeability Sieve (B3DB)
  Stage 2: Targeted P-glycoprotein Active Efflux Gatekeeper (Broccatelli)
"""

import os
import sys
import argparse
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.sequential_engine import SequentialCascadedEngine, BENCHMARK_DRUGS


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def print_banner():
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}     NOVELTY 2: SEQUENTIAL CASCADED SIEVE & TRANSFER ARCHITECTURE     {Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")


def display_result(res, name: Optional[str] = None):
    print_banner()
    if name:
        print(f"{Colors.BOLD}Compound Name:{Colors.ENDC} {name}")
    print(f"{Colors.BLUE}Input SMILES :{Colors.ENDC} {res.smiles}\n")

    print(f"{Colors.BOLD}{Colors.UNDERLINE}STAGE 1: BLOOD-BRAIN BARRIER (BBB) PERMEATION SIEVE{Colors.ENDC}")
    s1_status = f"{Colors.GREEN}PASSED (Permeant){Colors.ENDC}" if res.stage1_pass else f"{Colors.RED}REJECTED (Non-permeant){Colors.ENDC}"
    print(f"Status           : {s1_status}")
    print(f"P(BBB Permeant)  : {res.stage1_prob_bbb*100:.1f}% (Threshold: {res.stage1_threshold*100:.1f}%)")
    print(f"Stage 1 Latency  : {res.stage1_latency_ms:.2f} ms\n")

    print(f"{Colors.BOLD}{Colors.UNDERLINE}STAGE 2: P-GLYCOPROTEIN (P-gp) ACTIVE EFFLUX GATEKEEPER{Colors.ENDC}")
    if not res.stage2_evaluated:
        print(f"{Colors.YELLOW}Status           : BYPASSED (Stage 1 Non-Permeant -> Stage 2 Skipped!){Colors.ENDC}")
        print(f"Compute Saved    : {Colors.GREEN}YES (Avoided 3D conformer embedding & ETKDGv3 overhead){Colors.ENDC}")
    else:
        s2_status = f"{Colors.GREEN}PASSED (Evades Efflux Pump){Colors.ENDC}" if res.stage2_pass else f"{Colors.RED}CAUGHT (Substrate / Active Efflux Liability){Colors.ENDC}"
        print(f"Status           : {s2_status}")
        print(f"P(P-gp Efflux)   : {res.stage2_prob_pgp*100:.1f}%")
        print(f"Stage 2 Latency  : {res.stage2_latency_ms:.2f} ms")

    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}FINAL CLINICAL DISPOSITION{Colors.ENDC}")
    if res.final_disposition == "CNS_PENETRANT":
        disp_colored = f"{Colors.GREEN}{Colors.BOLD}CNS_PENETRANT (Optimal Central Action){Colors.ENDC}"
    elif res.final_disposition == "PARADOXICAL_EFFLUX":
        disp_colored = f"{Colors.RED}{Colors.BOLD}PARADOXICAL_EFFLUX (Permeates Membrane, But Extruded By Pump!){Colors.ENDC}"
    else:
        disp_colored = f"{Colors.YELLOW}{Colors.BOLD}PERIPHERAL_LOW_PERMEATION (Action Confined to Periphery){Colors.ENDC}"

    print(f"Disposition      : {disp_colored}")
    print(f"Cascaded S_CNS   : {res.composite_cns_score:.4f}")
    print(f"Total Latency    : {res.total_latency_ms:.2f} ms")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def run_benchmarks(engine: SequentialCascadedEngine):
    print_banner()
    print(f"{Colors.BOLD}EVALUATING 5 CLINICAL BENCHMARK DRUGS THROUGH THE SEQUENTIAL SIEVE:{Colors.ENDC}\n")
    for name, data in BENCHMARK_DRUGS.items():
        res = engine.evaluate_molecule(data["smiles"])
        display_result(res, name=name)


def main():
    parser = argparse.ArgumentParser(description="Novelty 2: Sequential Cascaded Sieve CLI")
    parser.add_argument("--smiles", type=str, help="SMILES string of target molecule")
    parser.add_argument("--test-benchmark", action="store_true", help="Run 5 clinical reference benchmark drugs")
    parser.add_argument("--tau-bbb", type=float, default=0.50, help="Stage 1 BBB probability threshold")
    parser.add_argument("--tau-pgp", type=float, default=0.50, help="Stage 2 P-gp efflux probability threshold")
    args = parser.parse_args()

    engine = SequentialCascadedEngine(tau_bbb=args.tau_bbb, tau_pgp=args.tau_pgp)

    if args.test_benchmark:
        run_benchmarks(engine)
    elif args.smiles:
        res = engine.evaluate_molecule(args.smiles)
        display_result(res)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
