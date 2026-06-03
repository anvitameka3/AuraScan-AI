import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import os
import csv

class BiophotonicGermNet(nn.Module):
    def __init__(self):
        super(BiophotonicGermNet, self).__init__()
        self.enc1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1  = nn.BatchNorm2d(16)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2  = nn.BatchNorm2d(32)
        self.dec1 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn3  = nn.BatchNorm2d(16)
        self.dec2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x = F.relu(self.bn3(self.dec1(x)))
        x = torch.sigmoid(self.dec2(x))
        return x

class QuantumOpticalProcessor:
    def __init__(self, height=480, width=640):
        self.height = height
        self.width = width
        self.prev_gray_float = None
        self.model = BiophotonicGermNet()
        self.model.eval()
        
    def extract_spatial_scattering(self, gray_img: np.ndarray) -> np.ndarray:
        f_transform = np.fft.fft2(gray_img.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        mask = np.ones((self.height, self.width), np.uint8)
        crow, ccol = self.height // 2, self.width // 2
        cv2.circle(mask, (ccol, crow), 45, 0, -1)
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_backscattering = np.abs(np.fft.ifft2(f_ishift))
        cv2.normalize(img_backscattering, img_backscattering, 0, 1, cv2.NORM_MINMAX)
        return img_backscattering

    def extract_metabolic_micro_flow(self, gray_img: np.ndarray) -> np.ndarray:
        gray_float = gray_img.astype(np.float32) / 255.0
        if self.prev_gray_float is None:
            self.prev_gray_float = gray_float
        frame_variance = cv2.absdiff(gray_float, self.prev_gray_float)
        micro_flicker = cv2.GaussianBlur(frame_variance, (9, 9), 0)
        cv2.normalize(micro_flicker, micro_flicker, 0, 1, cv2.NORM_MINMAX)
        self.prev_gray_float = gray_float
        return micro_flicker

    def execute_live_pipeline(self):
        cap = None
        
        # SYSTEM PROBE: Auto-scan camera ports to find the native hardware stream
        for idx in range(3):
            print(f"[SYSTEM] Probing Camera Hardware Link Index: {idx}...")
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            time.sleep(1.0)
            if cap.isOpened():
                ret, test_frame = cap.read()
                # Ensure the camera is returning a valid, non-black video canvas frame
                if ret and test_frame is not None and np.sum(test_frame) > 0:
                    print(f"[SUCCESS] Native live feed initialized on port index: {idx}!")
                    break
            cap.release()
            cap = None

        if cap is None:
            print("\n[CRITICAL ERROR] Camera stream could not be opened by the system.")
            print("[HELP] Go to System Settings -> Privacy & Security -> Camera, and confirm Terminal is toggled ON.\n")
            sys.exit(1)
            
        print("\n=======================================================")
        print("  AURASCAN AI: LIVE CAM TRACKING MECHANICS INITIALIZED ")
        print("=======================================================")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv2.namedWindow("AuraScan AI - Live Pathogen Detection Dashboard", cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            frame = cv2.resize(frame, (self.width, self.height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            spatial_scattering = self.extract_spatial_scattering(gray)
            metabolic_flow = self.extract_metabolic_micro_flow(gray)
            
            input_tensor = np.stack([spatial_scattering, metabolic_flow], axis=0)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float()
            
            with torch.no_grad():
                prob_map = self.model(input_tensor).squeeze().numpy()
                
            germ_mask = (prob_map > 0.45).astype(np.uint8)
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(germ_mask)
            germs_found = max(0, num_labels - 1)
            
            contamination_ratio = np.sum(germ_mask > 0) / germ_mask.size
            sterility_score = round(100.0 * np.exp(-1.5 * contamination_ratio), 2)
            
            heatmap_overlay = np.zeros_like(frame)
            heatmap_overlay[:, :, 1] = (prob_map * 255).astype(np.uint8)
            heatmap_overlay = cv2.GaussianBlur(heatmap_overlay, (15, 15), 0)
            composite_overlay = cv2.addWeighted(frame, 0.75, heatmap_overlay, 1.2, 0)
            
            hud_text = f"LIVE INDEX: {sterility_score}%"
            hud_color = (0, 255, 0) if sterility_score > 90.0 else (0, 255, 255) if sterility_score > 75.0 else (0, 0, 255)
            
            cv2.putText(composite_overlay, hud_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
            cv2.putText(composite_overlay, f"LIVE TARGET CLUSTERS: {germs_found}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(composite_overlay, "FEED: LIVE FACE-TIME CAM INTERACTIVE", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
            dashboard_panel = np.hstack((frame, composite_overlay))
            cv2.imshow("AuraScan AI - Live Pathogen Detection Dashboard", dashboard_panel)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = QuantumOpticalProcessor()
    detector.execute_live_pipeline()
