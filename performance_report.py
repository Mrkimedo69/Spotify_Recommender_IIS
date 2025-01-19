import re
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

LOG_PERFORMANCE_PATH = "logs/performance.log"
REPORT_OUTPUT_PATH = "reports/performance_report.pdf"

def parse_performance_log(log_path):
    """
    Parse the performance log file to extract performance data.
    """
    pattern = r"^\[(.*?)\] Version: (\S+), Precision: (\S+), Description: (.+)$"
    entries = []

    with open(log_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                timestamp, version, precision, description = match.groups()
                entries.append({
                    "timestamp": timestamp,
                    "version": version,
                    "precision": float(precision),
                    "description": description
                })
            else:
                print(f"Skipping malformed line: {line.strip()}")

    if entries:
        return pd.DataFrame(entries)
    else:
        print("No performance data available to parse.")
        return pd.DataFrame()

def generate_performance_report(log_path, output_path):
    """
    Generate a PDF report of model performance with a plot.
    """
    performance_data = parse_performance_log(log_path)

    if performance_data.empty:
        print("No performance data available to generate a report.")
        return

    # Generate a plot of performance over versions
    versions = performance_data['version']
    precision = performance_data['precision']

    plt.figure(figsize=(8, 5))
    plt.plot(versions, precision, marker='o', label='Precision')
    for i, txt in enumerate(precision):
        plt.annotate(f"{txt:.2f}", (versions[i], precision[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title("Model Performance Over Versions")
    plt.xlabel("Model Version")
    plt.ylabel("Precision")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plot_path = output_path.replace(".pdf", ".png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved: {plot_path}")

    # Generate a PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Model Performance Report", ln=True, align='C')
    pdf.ln(10)

    # Add data to PDF
    for index, entry in performance_data.iterrows():
        pdf.cell(0, 10, f"Version: {entry['version']} | Precision: {entry['precision']:.2f}", ln=True)
        pdf.cell(0, 10, f"Description: {entry['description']}", ln=True)
        pdf.cell(0, 10, f"Timestamp: {entry['timestamp']}", ln=True)
        pdf.ln(5)

    # Add the plot to the PDF
    pdf.add_page()
    pdf.cell(0, 10, "Performance Plot", ln=True, align='C')
    pdf.image(plot_path, x=10, y=30, w=190)

    pdf.output(output_path)
    print(f"Performance report generated: {output_path}")

if __name__ == "__main__":
    generate_performance_report(LOG_PERFORMANCE_PATH, REPORT_OUTPUT_PATH)
