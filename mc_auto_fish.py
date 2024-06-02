import cv2
import numpy as np
import pyautogui
import tkinter as tk
import time
from prettytable import PrettyTable
import statistics
import os


# Set up the path for the template image
def get_image_path(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, filename)


# Define the Tkinter window to select the region for template matching
def select_region():
    def on_confirm():
        global selected_region
        x1 = root.winfo_rootx()
        y1 = root.winfo_rooty()
        x2 = x1 + root.winfo_width()
        y2 = y1 + root.winfo_height()
        selected_region = (x1, y1, x2 - x1, y2 - y1)
        root.after(2000, root.destroy)

    root = tk.Tk()
    root.title("Select Region")
    root.geometry('400x280')
    root.wm_attributes('-alpha', 0.8)
    root.wm_attributes('-topmost', True)

    confirm_button = tk.Button(root,
                               text="Adjust the window to cover the subtitles\nClick anywhere inside the window\nto confirm the region",
                               command=on_confirm)
    confirm_button.pack(fill=tk.BOTH, expand=True)

    root.mainloop()
    return selected_region


# Perform multi-scale template matching
def multi_scale_template_matching(image_path, region):
    template = cv2.imread(image_path, 0)
    x, y, w, h = region
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    best_match = None
    max_val = 0
    scales = np.linspace(0.7, 2, 50)[::-1]

    for scale in scales:
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        if resized_template.shape[0] > h or resized_template.shape[1] > w:
            continue

        res = cv2.matchTemplate(screenshot_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val_temp, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val_temp > max_val:
            max_val = max_val_temp
            best_match = max_loc, scale

    if best_match and max_val > 0.75:
        max_loc, scale = best_match
        print(f"Match found. Scale: {scale:.6f}, Location: {max_loc}, Similarity: {max_val:.6f}")
        return True, scale
    else:
        print(f"No match found. Highest similarity: {max_val:.6f}")
        return False, None


# Run the matching process in a loop and perform actions based on the results
def run_matching_loop(image_path, selected_region):
    start_time = time.time()
    match_count = 0
    match_times = []
    last_match_time = time.time()
    no_match_start_time = time.time()

    while True:
        match_found, scale = multi_scale_template_matching(image_path, selected_region)
        current_time = time.time()

        if match_found:
            match_count += 1
            match_interval = current_time - last_match_time
            match_times.append(match_interval)
            last_match_time = current_time
            no_match_start_time = current_time

            table = PrettyTable()
            table.header = False
            table.add_row(["Match Count", match_count])
            table.add_row(["Time Since Last Match", f"{match_interval:.4f} s"])
            table.add_row(["Min Time", f"{min(match_times):.4f} s"])
            table.add_row(["Max Time", f"{max(match_times):.4f} s"])
            table.add_row(["Median Time", f"{statistics.median(match_times):.4f} s"])
            table.add_row(["Average Time", f"{sum(match_times) / len(match_times):.4f} s"])

            print(table)

            pyautogui.click(button='right')
            time.sleep(1)
            pyautogui.click(button='right')
            time.sleep(3)
        else:
            if current_time - no_match_start_time >= 40:
                print("No match found for 40 seconds, attempting to return to the game and perform a right-click")
                pyautogui.press('tab')
                pyautogui.press('enter')
                pyautogui.click(button='right')
                no_match_start_time = current_time

        time.sleep(0.4)


# Main function to execute the workflow
def main():
    image_path = get_image_path('target_eg.png')
    print(f"Image for matching located at: {image_path}")

    selected_region = select_region()
    if selected_region:
        print(f"Selected region: {selected_region}")
        run_matching_loop(image_path, selected_region)
    else:
        print("No region selected")


if __name__ == "__main__":
    main()
