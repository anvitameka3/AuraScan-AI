import cv2
import numpy as np
import sys
import time
import os
import csv

class QuantumOpticalProcessor:
    def __init__(self, height=480, width=640):
        self.height = height
        self.width = width
        
    def generate_clean_hand_matrix(self) -> tuple:
        """
        Simulates a post-wash hand matrix profile where biological 
        contamination has been mitigated by 90%.
        """
        canvas = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        cv2.ellipse(canvas, (320, 360), (120, 160), 0, 0, 360, (210, 180, 140), -1) 
        cv2.ellipse(canvas, (240, 160), (22, 90), -15, 0, 360, (210, 180, 140), -1)  
        cv2.ellipse(canvas, (310, 120), (24, 100), 0, 0, 360, (210, 180, 140), -1)   
        cv2.ellipse(canvas, (380, 150), (22, 95), 10, 0, 360, (210, 180, 140), -1)   
        cv2.ellipse(canvas, (450, 220), (20, 75), 25, 0, 360, (210, 180, 140), -1)   
        cv2.ellipse(canvas, (180, 340), (25, 70), -45, 0, 360, (210, 180, 140), -1)  

        hand_mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, hand_binary = cv2.threshold(hand_mask, 50, 255, cv2.THRESH_BINARY)

        # Shrink germ sizes down to small micro-traces
        germ_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(germ_layer, (310, 140), 2, 255, -1)  
        cv2.circle(germ_layer, (320, 320), 4, 255, -1)  
        cv2.circle(germ_layer, (230, 200), 3, 255, -1)  
        
        germ_layer = cv2.bitwise_and(germ_layer, hand_binary)
        
        # FIXED: Explicitly set the trace germ pixels to a dark organic tone
        canvas[germ_layer > 0] = [80, 140, 100]
        
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        return canvas, germ_layer

    def append_to_csv_log(self, score, clusters, status):
        csv_file = "aurascan_history_log.csv"
        file_exists = os.path.isfile(csv_file)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Sterility Score (%)", "Pathogen Hotspots Found", "Bio-Safety Status"])
            writer.writerow([timestamp, score, clusters, status])

    def generate_html_report(self, score, clusters):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        if score > 90.0:
            status_tier = "OPTIMAL / STERILE"
            status_color = "#2ed573"
            advice = "The hand surface matches medical baseline safety standards. Maintain regular sanitization protocols."
        elif score > 75.0:
            status_tier = "MODERATE CONTAMINATION"
            status_color = "#ffa502"
            advice = "Localized microbial activity detected. A standard 20-second soap wash is advised."
        else:
            status_tier = "CRITICAL PATHOGEN DETECTED"
            status_color = "#ff4757"
            advice = "Significant surface clusters identified. Immediate sanitation cycle recommended."

        self.append_to_csv_log(score, clusters, status_tier)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AuraScan AI - Diagnostic Report</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #0f1115; color: #e1e7f0; margin: 0; padding: 40px; }}
        .card {{ max-width: 700px; margin: 0 auto; background: #161a22; border-radius: 16px; padding: 40px; border: 1px solid #262d3d; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        h1 {{ color: #ffffff; font-size: 28px; margin-top: 0; border-bottom: 2px solid #262d3d; padding-bottom: 15px; letter-spacing: 1px; }}
        .metric-group {{ display: flex; justify-content: space-between; margin: 30px 0; }}
        .metric-box {{ background: #1e2430; border-radius: 8px; padding: 20px; width: 45%; border: 1px solid #2e374a; text-align: center; }}
        .value {{ font-size: 32px; font-weight: bold; margin-top: 10px; }}
        .status-banner {{ background: {status_color}20; border: 1px solid {status_color}; color: {status_color}; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px; text-transform: uppercase; margin: 20px 0; letter-spacing: 1px; }}
        .guidance {{ background: #1c1f26; border-left: 4px solid #57606f; padding: 20px; border-radius: 0 8px 8px 0; font-size: 15px; line-height: 1.6; color: #a4b0be; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>AURASCAN AI // BIOPHOTONIC ANALYSIS REPORT</h1>
        <p style="color: #747d8c; font-size: 14px;"><strong>Timestamp:</strong> {timestamp}</p>
        <div class="status-banner">{status_tier}</div>
        <div class="metric-group">
            <div class="metric-box">
                <div style="color: #747d8c; font-size: 13px; text-transform: uppercase; font-weight: bold;">Sterility Rating</div>
                <div class="value" style="color: {status_color};">{score}%</div>
            </div>
            <div class="metric-box">
                <div style="color: #747d8c; font-size: 13px; text-transform: uppercase; font-weight: bold;">Pathogen Hotspots</div>
                <div class="value" style="color: #ffffff;">{clusters}</div>
            </div>
        </div>
        <h3>Clinical Hygiene Guidance</h3>
        <div class="guidance">{advice}</div>
    </div>
</body>
</html>"""
        with open("aurascan_diagnostic_report.html", "w") as f:
            f.write(html_content)

    def execute_simulation_pipeline(self):
        frame, true_germ_mask = self.generate_clean_hand_matrix()
        prob_map = true_germ_mask.astype(np.float32) / 255.0
        
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(true_germ_mask)
        germs_found = max(0, num_labels - 1)
        
        contamination_ratio = np.sum(true_germ_mask > 0) / true_germ_mask.size
        sterility_score = round(100.0 * np.exp(-35.0 * contamination_ratio), 2)
        
        heatmap_overlay = np.zeros_like(frame)
        heatmap_overlay[:, :, 1] = (prob_map * 255).astype(np.uint8)
        heatmap_overlay = cv2.GaussianBlur(heatmap_overlay, (5, 5), 0)
        composite_overlay = cv2.addWeighted(frame, 0.80, heatmap_overlay, 1.0, 0)
        
        hud_text = f"STERILITY INDEX: {sterility_score}%"
        hud_color = (0, 255, 0) if sterility_score > 90.0 else (0, 255, 255) if sterility_score > 75.0 else (0, 0, 255)
        
        cv2.putText(composite_overlay, hud_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
        cv2.putText(composite_overlay, f"PATHOGEN CLUSTERS LOCATED: {germs_found}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        dashboard_panel = np.hstack((frame, composite_overlay))
        self.generate_html_report(sterility_score, germs_found)
        
        while True:
            cv2.imshow("AuraScan AI - Pathogen Simulation Dashboard", dashboard_panel)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = QuantumOpticalProcessor()
    detector.execute_pipeline = detector.execute_simulation_pipeline
    detector.execute_pipeline()
