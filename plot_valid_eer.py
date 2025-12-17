"""
Plot validation EER from all experiments for paper comparison

Extracts 'val/eer' metrics from TensorBoard logs and creates a publication-ready figure.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
matplotlib.rcParams['ps.fonttype'] = 42


def extract_eer_from_tb(log_dir):
    """
    Extract validation EER from TensorBoard log directory
    
    Args:
        log_dir: Path to TensorBoard log directory (e.g., logs/exp1_basic/version_0)
    
    Returns:
        steps: List of step numbers
        eers: List of EER values (as percentages)
    """
    # Find event file
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None, None
    
    # Use the first (and usually only) event file
    event_file = event_files[0]
    
    try:
        # Load event accumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Get scalar events for 'val/eer'
        if 'val/eer' not in ea.Tags()['scalars']:
            print(f"Warning: 'val/eer' not found in {log_dir}")
            return None, None
        
        scalar_events = ea.Scalars('val/eer')
        
        # Extract steps and values
        steps = [s.step for s in scalar_events]
        eers = [s.value * 100 for s in scalar_events]  # Convert to percentage
        
        return steps, eers
    
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
        return None, None


def get_all_experiments(logs_dir):
    """
    Get all experiment directories from logs folder
    
    Returns:
        List of (experiment_name, version_dirs) tuples
    """
    experiments = []
    
    # Get all experiment directories
    for exp_dir in sorted(os.listdir(logs_dir)):
        exp_path = os.path.join(logs_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # Get all version directories
        version_dirs = []
        for version_dir in sorted(os.listdir(exp_path)):
            version_path = os.path.join(exp_path, version_dir)
            if os.path.isdir(version_path):
                version_dirs.append(version_path)
        
        if version_dirs:
            experiments.append((exp_dir, version_dirs))
    
    return experiments


def plot_all_experiments(logs_dir, output_path='logs/image/valid_eer_comparison.pdf', 
                        figsize=(10, 6), max_eer=None, exclude_high_eer=True):
    """
    Plot validation EER for all experiments
    
    Args:
        logs_dir: Path to logs directory
        output_path: Output file path for the figure
        figsize: Figure size (width, height)
        max_eer: Maximum EER to display (for y-axis limit)
        exclude_high_eer: If True, exclude experiments with final EER > 10%
    """
    experiments = get_all_experiments(logs_dir)
    
    if not experiments:
        print(f"No experiments found in {logs_dir}")
        return
    
    # Create figure with publication-ready style
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color scheme - use distinct colors for better visibility
    # Use a palette that works well in both color and grayscale
    color_list = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    # Collect all data first
    plot_data = []
    for idx, (exp_name, version_dirs) in enumerate(experiments):
        # Try all versions and use the one with most data points
        best_steps = None
        best_eers = None
        best_version = None
        
        for version_dir in version_dirs:
            steps, eers = extract_eer_from_tb(version_dir)
            if steps is not None and eers is not None:
                if best_steps is None or len(steps) > len(best_steps):
                    best_steps = steps
                    best_eers = eers
                    best_version = version_dir
        
        if best_steps is not None and best_eers is not None:
            final_eer = best_eers[-1]
            # Skip if exclude_high_eer and final EER is too high
            if exclude_high_eer and final_eer > 10.0:
                print(f"{exp_name:20s} | Final EER: {final_eer:.4f}% | EXCLUDED (too high)")
                continue
            
            plot_data.append({
                'name': exp_name,
                'steps': best_steps,
                'eers': best_eers,
                'final_eer': final_eer,
                'color': color_list[idx % len(color_list)]
            })
            print(f"{exp_name:20s} | Final EER: {final_eer:.4f}% | Steps: {len(best_steps)}")
        else:
            print(f"{exp_name:20s} | No valid EER data found")
    
    # Sort by final EER for consistent ordering
    plot_data.sort(key=lambda x: x['final_eer'])
    
    # Plot each experiment
    for data in plot_data:
        # Clean experiment name for display
        display_name = data['name'].replace('_', ' ').title()
        if 'Exp' in display_name:
            # Format: Exp1 Basic -> Exp 1 (Basic)
            parts = display_name.split()
            if len(parts) >= 2:
                exp_num = parts[0]
                # exp_type = ' '.join(parts[1:])
                display_name = f"{exp_num}"
        
        # Plot line
        ax.plot(data['steps'], data['eers'], 
               label=display_name, 
               color=data['color'],
               linewidth=2.5,
               alpha=0.85)
        
        # Mark final EER
        final_eer = data['final_eer']
        final_step = data['steps'][-1]
        ax.scatter([final_step], [final_eer], 
                  color=data['color'], 
                  s=80, 
                  zorder=5,
                  edgecolors='white',
                  linewidths=1.5,
                  marker='o')
    
    # Formatting
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation EER (%)', fontsize=13, fontweight='bold')
    ax.set_title('Validation EER Comparison Across Experiments', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', fontsize=10, framealpha=0.95, 
             fancybox=True, shadow=True, ncol=1)
    
    # Set y-axis limits
    if max_eer is not None:
        ax.set_ylim(bottom=0, top=max_eer)
    else:
        # Auto-adjust but ensure it starts from 0
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=0, top=max(y_max * 1.05, 5.0))
    
    # Improve layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # # Save figure as PDF
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"\nFigure saved to: {output_path}")
    
    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {png_path}")
    
    plt.close()


def create_summary_table(logs_dir, output_path='valid_eer_summary.txt'):
    """
    Create a summary table of final EER values
    
    Args:
        logs_dir: Path to logs directory
        output_path: Output file path for summary
    """
    experiments = get_all_experiments(logs_dir)
    
    results = []
    
    for exp_name, version_dirs in experiments:
        best_final_eer = None
        best_version = None
        
        for version_dir in version_dirs:
            steps, eers = extract_eer_from_tb(version_dir)
            if steps is not None and eers is not None:
                final_eer = eers[-1]
                if best_final_eer is None or final_eer < best_final_eer:
                    best_final_eer = final_eer
                    best_version = version_dir
        
        if best_final_eer is not None:
            results.append((exp_name, best_final_eer, len(steps) if steps else 0))
    
    # Sort by EER (ascending)
    results.sort(key=lambda x: x[1])
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Write summary
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Validation EER Summary (Sorted by Final EER)\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Experiment':<25} {'Final EER (%)':<15} {'Data Points':<15}\n")
        f.write("-"*80 + "\n")
        
        for exp_name, final_eer, num_points in results:
            f.write(f"{exp_name:<25} {final_eer:>10.4f}%    {num_points:>10}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot validation EER from all TensorBoard logs'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='logs',
        help='Path to logs directory (default: logs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='logs/image/valid_eer_comparison.pdf',
        help='Output figure path (default: logs/image/valid_eer_comparison.pdf)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[10, 6],
        help='Figure size: width height (default: 10 6)'
    )
    parser.add_argument(
        '--max-eer',
        type=float,
        default=None,
        help='Maximum EER to display on y-axis (default: auto)'
    )
    parser.add_argument(
        '--include-high-eer',
        action='store_true',
        help='Include experiments with very high EER (>10%%)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        default=False,
        help='Also create summary table'
    )
    
    args = parser.parse_args()
    
    # Convert logs_dir to absolute path
    logs_dir = os.path.abspath(args.logs_dir)
    
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory not found: {logs_dir}")
        exit(1)
    
    print("="*80)
    print("Extracting Validation EER from TensorBoard Logs")
    print("="*80)
    print(f"Logs directory: {logs_dir}\n")
    
    # Create plot
    plot_all_experiments(logs_dir, args.output, tuple(args.figsize), 
                        max_eer=args.max_eer, 
                        exclude_high_eer=not args.include_high_eer)
    
    # Create summary if requested
    if args.summary:
        summary_path = args.output.replace('.pdf', '_summary.txt')
        create_summary_table(logs_dir, summary_path)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

