"""KG statistics: node/edge counts, degree distribution, power-law fit, etc."""

import pickle
import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter


def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def degree_stats(G: nx.Graph) -> dict:
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return {}
    arr = np.array(degrees)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "isolated_nodes": int(np.sum(arr == 0)),
    }


def power_law_fit(G: nx.Graph) -> dict:
    """Fit power-law to degree distribution using powerlaw package."""
    degrees = [d for _, d in G.degree() if d > 0]
    if len(degrees) < 10:
        return {"fit": False, "reason": "too few nodes"}
    try:
        import powerlaw
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        # Compare power-law vs exponential
        R, p = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
        return {
            "fit": True,
            "alpha": round(fit.alpha, 4),
            "xmin": float(fit.xmin),
            "loglikelihood_ratio_vs_exp": round(R, 4),
            "p_value": round(p, 4),
            "is_power_law": R > 0 and p < 0.05,
        }
    except ImportError:
        # Fallback: simple log-log linear regression
        freq = Counter(degrees)
        x = np.log10(sorted(freq.keys()))
        y = np.log10([freq[k] for k in sorted(freq.keys())])
        if len(x) < 3:
            return {"fit": False, "reason": "too few distinct degrees"}
        coeffs = np.polyfit(x, y, 1)
        return {
            "fit": True,
            "alpha_approx": round(-coeffs[0], 4),
            "r_squared": round(1 - np.sum((y - np.polyval(coeffs, x)) ** 2) / np.sum((y - np.mean(y)) ** 2), 4),
            "note": "install 'powerlaw' package for rigorous test",
        }


def component_stats(G: nx.Graph) -> dict:
    components = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in components], reverse=True)
    return {
        "num_components": len(components),
        "largest": sizes[0] if sizes else 0,
        "smallest": sizes[-1] if sizes else 0,
        "sizes_top5": sizes[:5],
    }


def graph_stats(G: nx.Graph) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    stats = {
        "nodes": n,
        "edges": m,
        "density": round(nx.density(G), 6),
        "avg_clustering": round(nx.average_clustering(G), 6) if n > 0 else 0,
        "degree": degree_stats(G),
        "components": component_stats(G),
        "power_law": power_law_fit(G),
    }
    # Diameter only if single component and small enough
    if stats["components"]["num_components"] == 1 and n <= 5000:
        stats["diameter"] = nx.diameter(G)
        stats["avg_shortest_path"] = round(nx.average_shortest_path_length(G), 4)
    return stats


def aggregate_stats(all_stats: list[dict]) -> dict:
    """Aggregate per-graph stats into dataset-level summary."""
    nodes = [s["nodes"] for s in all_stats]
    edges = [s["edges"] for s in all_stats]
    densities = [s["density"] for s in all_stats]
    clusterings = [s["avg_clustering"] for s in all_stats]
    pw = [s["power_law"].get("is_power_law") for s in all_stats if s["power_law"].get("fit")]

    return {
        "num_graphs": len(all_stats),
        "total_nodes": sum(nodes),
        "total_edges": sum(edges),
        "avg_nodes": round(np.mean(nodes), 2),
        "avg_edges": round(np.mean(edges), 2),
        "avg_density": round(np.mean(densities), 6),
        "avg_clustering": round(np.mean(clusterings), 6),
        "avg_degree_mean": round(np.mean([s["degree"]["mean"] for s in all_stats if s["degree"]]), 4),
        "power_law_count": sum(1 for v in pw if v),
        "power_law_total_tested": len(pw),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute KG statistics")
    parser.add_argument("--kg_dir", type=str, required=True,
                        help="Directory containing .pkl graph files (e.g. data/preprocessed/model/dataset/full_kg)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: print to stdout)")
    parser.add_argument("--start_index", type=int, default=None,
                        help="Start index (inclusive), filter by filename like dataset_IDX.pkl")
    parser.add_argument("--end_index", type=int, default=None,
                        help="End index (exclusive)")
    args = parser.parse_args()

    kg_dir = Path(args.kg_dir)
    pkl_files = sorted(kg_dir.glob("*.pkl"))

    # Filter by index range if specified
    if args.start_index is not None or args.end_index is not None:
        def _extract_idx(p: Path) -> int:
            # e.g. "legal_42.pkl" -> 42
            try:
                return int(p.stem.rsplit("_", 1)[-1])
            except ValueError:
                return -1
        lo = args.start_index if args.start_index is not None else 0
        hi = args.end_index if args.end_index is not None else float("inf")
        pkl_files = [p for p in pkl_files if lo <= _extract_idx(p) < hi]
    if not pkl_files:
        print(f"No .pkl files found in {kg_dir}")
        return

    print(f"Found {len(pkl_files)} graphs in {kg_dir}")

    all_stats = []
    for p in pkl_files:
        G = load_graph(str(p))
        s = graph_stats(G)
        s["file"] = p.name
        all_stats.append(s)
        print(f"  {p.name}: {s['nodes']} nodes, {s['edges']} edges, "
              f"density={s['density']}, clustering={s['avg_clustering']}")

    agg = aggregate_stats(all_stats)
    result = {"aggregate": agg, "per_graph": all_stats}

    print(f"\n{'='*60}")
    print(f"Aggregate: {agg['num_graphs']} graphs, "
          f"{agg['total_nodes']} total nodes, {agg['total_edges']} total edges")
    print(f"Avg nodes={agg['avg_nodes']}, Avg edges={agg['avg_edges']}, "
          f"Avg density={agg['avg_density']}, Avg clustering={agg['avg_clustering']}")
    print(f"Avg degree mean={agg['avg_degree_mean']}")
    print(f"Power-law: {agg['power_law_count']}/{agg['power_law_total_tested']} graphs")
    print(f"{'='*60}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {out}")
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
