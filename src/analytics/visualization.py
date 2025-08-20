import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, Any, List, Optional

from src.io import load_csv
from src.logging_config import get_logger

_logger = get_logger("visualization")

def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create visualizations based on configuration.
    
    Config parameters:
    - input_csv: Path to input CSV file
    - output_dir: Directory to save visualizations
    - visualization_type: Type of visualization ('comparative', 'statistical', 'radar')
    - metric_name: Name of metric to visualize (for comparative and statistical)
    - group_by: Column to group by (default: 'model')
    - hue: Column to use for color differentiation (default: 'prompt')
    - theme: Visual theme (default: 'academic')
    
    Returns:
    - Dict with keys:
      - success: Boolean indicating success/failure
      - artifacts: List of visualization files created
      - error: Error message (if success is False)
    """
    # Check required config keys
    required_keys = ["input_csv", "output_dir", "visualization_type"]
    for key in required_keys:
        if key not in cfg:
            _logger.error(f"missing config key: {key}")
            return {"success": False, "error": f"missing config key: {key}"}
    
    # Load input CSV
    try:
        _logger.info(f"loading input_csv: {cfg['input_csv']}")
        data = load_csv(cfg["input_csv"])
    except Exception as e:
        _logger.error(f"failed to load input_csv: {e}")
        return {"success": False, "error": f"failed to load input_csv: {e}"}
    
    # Verify data has required columns
    if len(data) == 0:
        _logger.error("input data is empty")
        return {"success": False, "error": "input data is empty"}
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Get visualization parameters
    viz_type = cfg["visualization_type"]
    metric_name = cfg.get("metric_name")
    group_by = cfg.get("group_by", "model")
    hue = cfg.get("hue", "prompt")
    theme = cfg.get("theme", "academic")
    
    # Create visualization instance
    viz = Visualization(theme=theme)
    artifacts = []
    
    try:
        if viz_type == "comparative":
            if not metric_name:
                _logger.error("metric_name is required for comparative visualization")
                return {"success": False, "error": "metric_name is required for comparative visualization"}
            
            output_path = os.path.join(cfg["output_dir"], f"comparative_{metric_name}.png")
            _logger.info(f"creating comparative visualization for {metric_name}")
            viz.plot_comparative_metrics(
                data=data.to_dict('records'),
                metric_name=metric_name,
                output_path=output_path
            )
            artifacts.append(output_path)
            
        elif viz_type == "statistical":
            if not metric_name:
                _logger.error("metric_name is required for statistical visualization")
                return {"success": False, "error": "metric_name is required for statistical visualization"}
            
            output_path = os.path.join(cfg["output_dir"], f"statistical_{metric_name}.png")
            _logger.info(f"creating statistical visualization for {metric_name}")
            viz.plot_statistical_visualizations(
                data=data.to_dict('records'),
                metric_name=metric_name,
                output_path=output_path
            )
            artifacts.append(output_path)
            
        elif viz_type == "radar":
            # For radar chart, we need to extract metrics and labels
            metrics_dict = cfg.get("metrics_dict", {})
            if not metrics_dict:
                # If metrics_dict not provided, try to extract from data
                numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if len(numeric_columns) == 0:
                    _logger.error("no numeric columns found for radar chart")
                    return {"success": False, "error": "no numeric columns found for radar chart"}
                
                # Use means of numeric columns as metrics
                metrics = [data[col].mean() for col in numeric_columns]
                labels = numeric_columns
            else:
                metrics = list(metrics_dict.values())
                labels = list(metrics_dict.keys())
            
            output_path = os.path.join(cfg["output_dir"], "radar_chart.png")
            _logger.info("creating radar chart")
            viz.plot_radar_chart(
                metrics=metrics,
                labels=labels,
                output_path=output_path
            )
            artifacts.append(output_path)
            
        else:
            _logger.error(f"unknown visualization_type: {viz_type}")
            return {"success": False, "error": f"unknown visualization_type: {viz_type}"}
        
        return {
            "success": True,
            "artifacts": artifacts
        }
        
    except Exception as e:
        _logger.error(f"error creating visualization: {e}")
        return {"success": False, "error": f"error creating visualization: {e}"}

def run(argv=None):
    """
    Run visualization from command line arguments.
    
    Arguments:
    - --input: Path to input CSV file
    - --output-dir: Directory to save visualizations
    - --type: Type of visualization ('comparative', 'statistical', 'radar')
    - --metric: Name of metric to visualize
    - --group-by: Column to group by (default: 'model')
    - --hue: Column to use for color differentiation (default: 'prompt')
    - --theme: Visual theme (default: 'academic')
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--type", required=True, choices=["comparative", "statistical", "radar"], 
                        help="Type of visualization")
    parser.add_argument("--metric", help="Name of metric to visualize")
    parser.add_argument("--group-by", default="model", help="Column to group by")
    parser.add_argument("--hue", default="prompt", help="Column to use for color differentiation")
    parser.add_argument("--theme", default="academic", help="Visual theme")
    
    args = parser.parse_args(argv)
    
    config = {
        "input_csv": args.input,
        "output_dir": args.output_dir,
        "visualization_type": args.type,
        "group_by": args.group_by,
        "hue": args.hue,
        "theme": args.theme
    }
    
    if args.metric:
        config["metric_name"] = args.metric
    
    result = run_from_config(config)
    if not result["success"]:
        print(f"ERROR: {result.get('error')}")
        sys.exit(1)
        # Early return to avoid accessing artifacts that don't exist in error case
        return result
    
    # Only reach this point for successful results
    print(f"Visualization(s) created: {', '.join(result['artifacts'])}")
    return result

# Keep the original Visualization class for backward compatibility
class Visualization:
    """Class for creating publication-ready visualizations."""

    def __init__(self, theme="academic"):
        self.theme = theme
        self._apply_theme()

    def _apply_theme(self):
        """Apply a consistent theme for all visualizations."""
        if self.theme == "academic":
            sns.set_theme(style="whitegrid", font_scale=1.2)
            plt.rcParams.update({
                "font.family": "serif",
                "figure.dpi": 300,
                "savefig.format": "png"
            })

    def plot_comparative_metrics(self, data, metric_name, output_path=None):
        """Create a comparative bar plot for a given metric."""
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="model", y=metric_name, hue="prompt", data=df)
        plt.title(f"Comparative Analysis of {metric_name}")
        plt.xlabel("Model")
        plt.ylabel(metric_name)
        plt.legend(title="Prompt")
        if output_path:
            plt.savefig(output_path)
        plt.close()  # Close the figure to avoid displaying it

    def plot_statistical_visualizations(self, data, metric_name, output_path=None):
        """Create statistical visualizations (box plots, violin plots)."""
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="model", y=metric_name, hue="prompt", data=df, split=True)
        plt.title(f"Statistical Visualization of {metric_name}")
        plt.xlabel("Model")
        plt.ylabel(metric_name)
        plt.legend(title="Prompt")
        if output_path:
            plt.savefig(output_path)
        plt.close()  # Close the figure to avoid displaying it

    def plot_radar_chart(self, metrics, labels, output_path=None):
        """Create a radar chart for multi-dimensional metrics."""
        import numpy as np
        from math import pi

        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels)
        ax.plot(angles, metrics + metrics[:1], linewidth=2, linestyle='solid')
        ax.fill(angles, metrics + metrics[:1], alpha=0.4)
        if output_path:
            plt.savefig(output_path)
        plt.close()  # Close the figure to avoid displaying it

    def export_figure(self, output_path, format="png"):
        """Export the last figure to a specified format."""
        plt.savefig(output_path, format=format)

if __name__ == "__main__":
    run()
