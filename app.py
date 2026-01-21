import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Snack Portion Planner", layout="centered")

# ======================
# SESSION STATE
# ======================
defaults = {
    "predicted": None,
    "gap": None,
    "total_consumed": 0.0,
    "option4_list": [],
    "option5_list": [],
    "option4_done": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    xls = pd.ExcelFile("Edited.xlsx")
    df_adults = pd.read_excel(xls, "Healthy_Adults_Meals")
    df_snacks = pd.read_excel(xls, "Snack_list_Cor")
    return df_adults, df_snacks

df_adults, df_snacks = load_data()

df_snacks = df_snacks.rename(columns={
    "Snack Type ": "Snack_Type",
    "Sub Type ": "Sub_Type",
    "Serving Size(g)/(ml) ": "Serving_Size",
    "g/ml": "Size_unit",
    "Car/g": "Carb_g",
    "Protein/g ": "Protein_g",
    "Fat/g": "Fat_g",
    "Energy/g (Kcal)": "Energy_kcal_g",
    "High energy/Medium enegy/Low enegy ": "Energy_Category",
    "Name ": "Name"
})

# ======================
# TRAIN MODEL
# ======================
@st.cache_resource
def train_model():
    X = df_adults[
        ['Gender', 'Age', 'Height (m)', 'Weight (kg)',
         'Daily_total_calory_intake_kcal', 'Income Level']
    ]
    y = df_adults['Total_Calory_Requirement_kcal_(BMR*PhysicalActivity)']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),
         ['Age', 'Height (m)', 'Weight (kg)', 'Daily_total_calory_intake_kcal']),
        ('cat', OneHotEncoder(drop='first'),
         ['Gender', 'Income Level'])
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('ridge', Ridge(alpha=1.0))
    ])

    model.fit(X, y)
    return model

model = train_model()

# ======================
# VISUALS
# ======================
def food_portion_atlas(row, max_items=5):
    items = np.arange(1, max_items + 1)
    energy = items * row["Serving_Size"] * row["Energy_kcal_g"]

    fig, ax = plt.subplots()
    ax.bar(items, energy)
    ax.set_xlabel("Number of Items")
    ax.set_ylabel("Energy (kcal)")
    ax.set_title(f"Food Portion Atlas: {row['Name']}")
    st.pyplot(fig)

def macro_pie_chart(row):
    macros = {
        "Carbs": row["Carb_g"],
        "Protein": row["Protein_g"],
        "Fat": row["Fat_g"]
    }
    fig, ax = plt.subplots()
    ax.pie(macros.values(), labels=macros.keys(), autopct="%1.1f%%")
    ax.axis("equal")
    st.pyplot(fig)

# ======================
# UI
# ======================
st.title("🍎 Personalized Snack Portion Planner")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 64, 25)
height = st.number_input("Height (m)", 1.0, 2.5, 1.65)
weight = st.number_input("Weight (kg)", 30.0, 150.0, 60.0)
daily_intake = st.number_input("Daily Calorie Intake (kcal)", 1000, 4000, 1800)
income = st.selectbox("Income Level", [1, 2, 3])

# ======================
# PREDICT
# ======================
if st.button("Predict Requirement"):
    bmi = weight / height**2

    if bmi < 18.5 or bmi > 22.9:
        st.error("Model valid only for healthy adults (BMI 18.5–22.9)")
        for k in defaults:
            st.session_state[k] = defaults[k]
    else:
        user = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Height (m)": height,
            "Weight (kg)": weight,
            "Daily_total_calory_intake_kcal": daily_intake,
            "Income Level": income
        }])

        pred = model.predict(user)[0]
        st.session_state.predicted = pred
        st.session_state.gap = pred - daily_intake
        st.session_state.total_consumed = 0
        st.session_state.option4_list.clear()
        st.session_state.option5_list.clear()
        st.session_state.option4_done = False

# ======================
# RESULTS
# ======================
if st.session_state.predicted is not None:
    remaining_gap = st.session_state.gap - st.session_state.total_consumed

    st.success(f"Predicted requirement: {st.session_state.predicted:.1f} kcal")
    st.info(f"Remaining gap: {remaining_gap:.1f} kcal")

    if remaining_gap <= 0:
        st.info("No additional snack needed.")

# ======================
# OPTIONS
# ======================
if st.session_state.predicted and remaining_gap > 0:

    option = st.radio(
        "Choose option:",
        (
            "1 - Portion size for a given snack",
            "2 - Suitability of eating a given number of items",
            "3 - One fixed snack + one suggested snack",
            "4 - Multiple snacks until energy fulfilled",
            "5 - Multiple snacks with fixed quantities"
        )
    )

    food = st.selectbox("Select Snack", df_snacks["Name"])
    row = df_snacks[df_snacks["Name"] == food].iloc[0]
    kcal_per_item = row["Serving_Size"] * row["Energy_kcal_g"]

    # -------- OPTION 1 --------
    if option.startswith("1"):
        portion = remaining_gap / row["Energy_kcal_g"]
        items = portion / row["Serving_Size"]
        st.write(f"Required portion: {portion:.1f} {row['Size_unit']}")
        st.write(f"Items: {items:.1f}")
        food_portion_atlas(row)
        macro_pie_chart(row)

    # -------- OPTION 2 --------
    elif option.startswith("2"):
        qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        energy = qty * kcal_per_item
        diff = remaining_gap - energy
        st.write(f"Energy: {energy:.1f} kcal")
        st.write(f"{'Remaining' if diff>=0 else 'Excess'}: {abs(diff):.1f}")
        food_portion_atlas(row)
        macro_pie_chart(row)

    # -------- OPTION 3 --------
    elif option.startswith("3"):
        qty1 = st.number_input("Items eaten (1st snack)", 0.0, 50.0, 1.0)
        energy1 = qty1 * kcal_per_item
        rem = remaining_gap - energy1
        st.write(f"{food}: {energy1:.1f} kcal")
        food_portion_atlas(row)
        macro_pie_chart(row)

        if rem > 0:
            food2 = st.selectbox("Second snack", df_snacks["Name"], key="sn2")
            row2 = df_snacks[df_snacks["Name"] == food2].iloc[0]
            portion2 = rem / row2["Energy_kcal_g"]
            items2 = portion2 / row2["Serving_Size"]
            st.write(f"{food2}: {items2:.1f} items")

    # -------- OPTION 4 --------
    elif option.startswith("4"):
        qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        if st.button("Add snack"):
            energy = min(qty * kcal_per_item, remaining_gap)
            st.session_state.total_consumed += energy
            st.session_state.option4_list.append((food, qty, energy))
            if st.session_state.total_consumed >= st.session_state.gap:
                st.session_state.option4_done = True

        for f, q, e in st.session_state.option4_list:
            st.write(f"{f}: {q} → {e:.1f} kcal")

        if st.session_state.option4_done:
            st.success("🎉 Requirement fulfilled")

    # -------- OPTION 5 --------
    elif option.startswith("5"):
        qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        if st.button("Add snack"):
            energy = qty * kcal_per_item
            st.session_state.option5_list.append((food, qty, energy))

        total = sum(e for _, _, e in st.session_state.option5_list)
        for f, q, e in st.session_state.option5_list:
            st.write(f"{f}: {q} → {e:.1f} kcal")

        st.info(f"Gap after snacks: {st.session_state.gap - total:.1f} kcal")
