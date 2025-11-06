#  Mental Health Self-Assessment Expert System (Tkinter)
# Simple rule-based expert system using if-elif-else statements

import tkinter as tk
from tkinter import messagebox

def get_result():
    sleep = sleep_var.get()
    stress = stress_var.get()
    motivation = motivation_var.get()
    social = social_var.get()

    if sleep == "" or stress == "" or motivation == "" or social == "":
        messagebox.showwarning("Incomplete", "Please answer all questions.")
        return

    # --- Rule base ---
    if sleep == "Yes" and stress == "Yes":
        result = ("Possible Cause: Stress or Sleep Deprivation\n\n"
                  "Tip: Practice relaxation and maintain a proper sleep routine.")
        messagebox.showerror("Mental Health Insight", result)
    elif stress == "Yes" and motivation == "Yes":
        result = ("Possible Cause: Mild Depression or Burnout\n\n"
                  "Tip: Take breaks, engage in hobbies, and talk to trusted friends.")
        messagebox.showwarning("Mental Health Insight", result)
    elif motivation == "Yes" and social == "Yes":
        result = ("Possible Cause: Social Withdrawal\n\n"
                  "Tip: Try outdoor or group activities to boost mood and confidence.")
        messagebox.showwarning("Mental Health Insight", result)
    elif sleep == "No" and stress == "No":
        result = ("You're doing great! \n\n"
                  "Maintain this healthy lifestyle and keep caring for your mind.")
        messagebox.showinfo("Mental Health Insight", result)
    else:
        result = ("Mixed or unclear symptoms.\n\n"
                  "Tip: Monitor your habits and consider consulting a counselor.")
        messagebox.showinfo("Mental Health Insight", result)

# --- GUI setup ---
root = tk.Tk()
root.title(" Mental Health Expert System")
root.geometry("500x500")
root.resizable(False, False)

title_label = tk.Label(root, text="Mental Health Self-Assessment Expert System",
                       font=("Helvetica", 14, "bold"), wraplength=450, pady=10)
title_label.pack()

# Question 1
sleep_var = tk.StringVar()
tk.Label(root, text="Do you sleep less than 6 hours daily?", font=("Arial", 11)).pack(anchor="w", padx=20, pady=5)
tk.Radiobutton(root, text="Yes", variable=sleep_var, value="Yes").pack(anchor="w", padx=40)
tk.Radiobutton(root, text="No", variable=sleep_var, value="No").pack(anchor="w", padx=40)

# Question 2
stress_var = tk.StringVar()
tk.Label(root, text="Do you often feel stressed or anxious?", font=("Arial", 11)).pack(anchor="w", padx=20, pady=5)
tk.Radiobutton(root, text="Yes", variable=stress_var, value="Yes").pack(anchor="w", padx=40)
tk.Radiobutton(root, text="No", variable=stress_var, value="No").pack(anchor="w", padx=40)

# Question 3
motivation_var = tk.StringVar()
tk.Label(root, text="Do you feel low motivation or interest in tasks?", font=("Arial", 11)).pack(anchor="w", padx=20, pady=5)
tk.Radiobutton(root, text="Yes", variable=motivation_var, value="Yes").pack(anchor="w", padx=40)
tk.Radiobutton(root, text="No", variable=motivation_var, value="No").pack(anchor="w", padx=40)

# Question 4
social_var = tk.StringVar()
tk.Label(root, text="Do you avoid social interactions or gatherings?", font=("Arial", 11)).pack(anchor="w", padx=20, pady=5)
tk.Radiobutton(root, text="Yes", variable=social_var, value="Yes").pack(anchor="w", padx=40)
tk.Radiobutton(root, text="No", variable=social_var, value="No").pack(anchor="w", padx=40)

# Button
tk.Button(root, text="🩺 Get Mental Health Insight", command=get_result,
          bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
          relief="raised", padx=10, pady=5).pack(pady=20)

# Footer
tk.Label(root, text="Made by [Your Name] | AIML Mini Project",
         font=("Arial", 9), fg="gray").pack(side="bottom", pady=10)

root.mainloop()
