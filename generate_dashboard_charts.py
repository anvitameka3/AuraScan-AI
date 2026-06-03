import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def render_hygiene_trends():
    csv_file = "aurascan_history_log.csv"
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] Target file '{csv_file}' not found. Run your detector app first to populate the database.")
        sys.exit(1)
        
    # Read persistent data logs
    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("[ERROR] The data log file is empty. No telemetry metrics available to plot.")
        return

    # Configure advanced aesthetic layout parameters
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Render line plot mapping trends over time
    ax.plot(df['Timestamp'], df['Sterility Score (%)'], marker='o', color='#2ed573', linewidth=2.5, markersize=8, label='Sterility Rating')
    
    # Format graph overlay elements
    ax.set_title('AuraScan AI - Bio-Safety Telemetry Timeline Analytics', fontsize=14, pad=20, weight='bold', color='#ffffff')
    ax.set_xlabel('Scan Execution Timestamp Log', fontsize=11, labelpad=10, color='#a4b0be')
    ax.set_ylabel('Sterility Rating Index Value (%)', fontsize=11, labelpad=10, color='#a4b0be')
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')
    
    # Auto-rotate timestamp layout axis tags cleanly to prevent pixel overlaps
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    # Save optimized graphic chart asset
    output_img = "aurascan_analytics_chart.png"
    plt.savefig(output_img, dpi=300)
    print(f"[SUCCESS] High-contrast chart compiled and saved as: {output_img}")
    
    # Display running window to user
    plt.show()

if __name__ == "__main__":
    # Dynamically inject pandas mapping if not explicitly declared in system profiles
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    
    render_hygiene_trends()
