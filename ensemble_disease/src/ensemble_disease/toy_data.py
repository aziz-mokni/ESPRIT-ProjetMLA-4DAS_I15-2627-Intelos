"""The two worked examples from the course annexes."""
import pandas as pd

GOLF = pd.DataFrame(
    [
        ("Sunny", "Hot", "High", "Weak", "No"), ("Sunny", "Hot", "High", "Strong", "No"),
        ("Overcast", "Hot", "High", "Weak", "Yes"), ("Rain", "Mild", "High", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
        ("Sunny", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Mild", "Normal", "Weak", "Yes"),
        ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
        ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No"),
    ],
    columns=["Outlook", "Temp", "Humidity", "Wind", "Decision"],
)

H, L = "High", "Low"
STUDENT = pd.DataFrame(
    [
        (H, H, H, H, "YES"), (H, H, L, L, "YES"), (L, L, H, L, "NO"), (H, L, L, H, "NO"),
        (L, L, H, H, "YES"), (H, H, L, H, "YES"), (L, H, H, H, "YES"), (L, L, L, H, "NO"),
        (H, L, L, L, "YES"), (L, L, L, L, "NO"), (L, H, L, L, "NO"), (H, H, H, L, "YES"),
        (H, L, H, H, "YES"), (L, H, H, L, "NO"),
    ],
    columns=["Attendance", "StudyTime", "PreviousGrade", "Homework", "Success"],
    index=[f"E{i}" for i in range(1, 15)],
)
