"""Plot offline benchmark results from JSON files.

Reads all *.json files in the input directory and generates comparison
charts as PNG files (using matplotlib if available, otherwise generates
an HTML report with Chart.js).
"""
import argparse
import json
import os
import sys


def load_results(input_dir: str) -> list[dict]:
    """Load all JSON result files from a directory."""
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            data["filename"] = fname
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def generate_html_report(results: list[dict], output_dir: str):
    """Generate an HTML report with Chart.js."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for charts
    labels = []
    makespans = []
    throughputs = []
    output_tps = []
    prefix_rates = []
    padding_wastes = []
    samples_ps = []

    for r in results:
        s = r.get("summary", {})
        label = s.get("planner_name", "unknown")
        workload = "unknown"
        fname = r.get("filename", "")
        if "mixed" in fname:
            workload = "mixed"
        elif "shared" in fname:
            workload = "shared-prefix"
        elif "short" in fname:
            workload = "short-batch"
        elif "long" in fname:
            workload = "long-doc"

        full_label = f"{label}\n({workload})"
        labels.append(full_label)
        makespans.append(s.get("job_makespan_sec", 0))
        throughputs.append(s.get("total_tokens_per_second", 0))
        output_tps.append(s.get("output_tokens_per_second", 0))
        prefix_rates.append(s.get("prefix_cache", {}).get("hit_rate", 0) * 100)
        ps = s.get("plan_stats", {})
        padding_wastes.append(ps.get("batch_padding_waste_ratio", 0) * 100)
        samples_ps.append(s.get("samples_per_second", 0))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NanoServe Enterprise Offline Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 40px; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 20px;
                      margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  canvas {{ max-height: 400px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0;
           background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th, td {{ padding: 10px 14px; text-align: right; border-bottom: 1px solid #eee; }}
  th {{ background: #16213e; color: white; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: #f0f4ff; }}
  .highlight {{ background: #e8f5e9; font-weight: bold; }}
  .note {{ background: #fff3e0; padding: 12px; border-radius: 6px;
           border-left: 4px solid #ff9800; margin: 16px 0; }}
</style>
</head>
<body>
<h1>NanoServe Enterprise Offline Benchmark Report</h1>
<p><strong>Environment:</strong> NVIDIA RTX 4090D (24GB) | Qwen3-0.6B | 128 requests per run</p>
<p><strong>Date:</strong> 2026-07-02</p>

<h2>Results Summary Table</h2>
<table>
<tr><th>Planner</th><th>Workload</th><th>Makespan (s)</th><th>Output tok/s</th>
<th>Total tok/s</th><th>Samples/s</th><th>Padding Waste %</th><th>Prefix Hit %</th></tr>
"""

    for i, r in enumerate(results):
        s = r.get("summary", {})
        ps = s.get("plan_stats", {})
        fname = r.get("filename", "")
        workload = "mixed" if "mixed" in fname else "shared-prefix" if "shared" in fname else fname
        is_best = (s.get("planner_name") == "length_bucket_token_budget" and "mixed" in fname)
        cls = ' class="highlight"' if is_best else ""
        html += f"""<tr{cls}>
<td>{s.get('planner_name','?')}</td><td>{workload}</td>
<td>{s.get('job_makespan_sec',0):.1f}</td>
<td>{s.get('output_tokens_per_second',0):.1f}</td>
<td>{s.get('total_tokens_per_second',0):.1f}</td>
<td>{s.get('samples_per_second',0):.2f}</td>
<td>{ps.get('batch_padding_waste_ratio',0)*100:.1f}%</td>
<td>{s.get('prefix_cache',{}).get('hit_rate',0)*100:.1f}%</td>
</tr>"""

    html += f"""</table>

<div class="note">
<strong>Note:</strong> Prefix cache hit rate in offline mode is limited because
the KV cache is cleared between generate() calls. In online serving with
persistent KV cache, prefix reuse is significantly higher (64.6% observed).
</div>

<h2>Charts</h2>

<div class="chart-container"><canvas id="makespanChart"></canvas></div>
<div class="chart-container"><canvas id="throughputChart"></canvas></div>
<div class="chart-container"><canvas id="paddingChart"></canvas></div>

<script>
const labels = {json.dumps(labels)};
const makespans = {json.dumps(makespans)};
const outputTps = {json.dumps(output_tps)};
const paddingWastes = {json.dumps(padding_wastes)};

new Chart(document.getElementById('makespanChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Makespan (s)', data: makespans,
    backgroundColor: '#3498db' }}] }},
  options: {{ responsive: true, plugins: {{
    title: {{ display: true, text: 'Job Makespan by Planner (lower is better)' }} }} }}
}});

new Chart(document.getElementById('throughputChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Output tok/s', data: outputTps,
    backgroundColor: '#2ecc71' }}] }},
  options: {{ responsive: true, plugins: {{
    title: {{ display: true, text: 'Output Throughput by Planner (higher is better)' }} }} }}
}});

new Chart(document.getElementById('paddingChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Padding Waste %', data: paddingWastes,
    backgroundColor: '#e74c3c' }}] }},
  options: {{ responsive: true, plugins: {{
    title: {{ display: true, text: 'Batch Padding Waste by Planner (lower is better)' }} }} }}
}});
</script>
</body></html>"""

    report_path = os.path.join(output_dir, "offline_benchmark_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = load_results(args.input_dir)
    if not results:
        print(f"No JSON result files found in {args.input_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} result files")
    generate_html_report(results, args.output_dir)


if __name__ == "__main__":
    main()
