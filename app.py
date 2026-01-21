import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Snack Portion Planner", layout="centered")

# ======================
# SESSION STATE
# ======================
if "predicted" not in st.session_state:
    st.session_state.predicted = None
if "gap" not in st.session_state:
    st.session_state.gap = None
if "total_consumed" not in st.session_state:
    st.session_state.total_consumed = 0
if "option4_list" not in st.session_state:
    st.session_state.option4_list = []
if "option4_done" not in st.session_state:
    st.session_state.option4_done = False
if "option5_list" not in st.session_state:
    st.session_state.option5_list = []

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),
             ['Age', 'Height (m)', 'Weight (kg)', 'Daily_total_calory_intake_kcal']),
            ('cat', OneHotEncoder(drop='first'),
             ['Gender', 'Income Level'])
        ]
    )

    model = Pipeline([
        ('prep', preprocessor),
        ('ridge', Ridge(alpha=1.0))
    ])

    model.fit(X, y)
    return model

model = train_model()

# ======================
# VISUAL FUNCTIONS
# ======================
def food_portion_atlas(row, max_items=5):
    s = row["Serving_Size"]
    e = row["Energy_kcal_g"]
    items = np.arange(1, max_items + 1)
    energy = items * s * e
    fig, ax = plt.subplots()
    ax.bar(items, energy)
    ax.set_xlabel("Number of Items")
    ax.set_ylabel("Energy (kcal)")
    ax.set_title(f"Food Portion Atlas: {row['Name']}")
    st.pyplot(fig)

def macro_pie_chart(row):
    macros = {
        "Carbohydrates (g)": row["Carb_g"],
        "Protein (g)": row["Protein_g"],
        "Fat (g)": row["Fat_g"]
    }
    fig, ax = plt.subplots()
    ax.pie(macros.values(), labels=macros.keys(), autopct="%1.1f%%", startangle=90)
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
# PREDICTION
# ======================
if st.button("Predict Requirement"):
    bmi = weight / (height ** 2)

    if bmi < 18.5 or bmi > 22.9:
        st.error("Model valid only for healthy adults (BMI 18.5–22.9).")
        st.session_state.predicted = None
    else:
        user = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height (m)': [height],
            'Weight (kg)': [weight],
            'Daily_total_calory_intake_kcal': [daily_intake],
            'Income Level': [income]
        })

        pred = model.predict(user)[0]
        st.session_state.predicted = pred
        st.session_state.gap = pred - daily_intake
        st.session_state.total_consumed = 0
        st.session_state.option4_list = []
        st.session_state.option4_done = False
        st.session_state.option5_list = []

# ======================
# DISPLAY
# ======================
if st.session_state.predicted is not None:
    st.success(f"Predicted Requirement: {st.session_state.predicted:.1f} kcal/day")
    st.info(f"Initial Calorie Gap: {st.session_state.gap:.1f} kcal")

# ======================
# OPTIONS MENU
# ======================
if st.session_state.predicted is not None:
    option = st.radio(
        "Choose snack option:",
        (
            "1 - Portion size for a given snack",
            "2 - Suitability of eating a given number of items",
            "3 - One fixed snack + one suggested snack",
            "4 - Multiple snacks until energy fulfilled",
            "5 - Multiple snacks with fixed quantities"
        )
    )

    food_name = st.selectbox("Select Snack", df_snacks["Name"].unique())
    row = df_snacks[df_snacks["Name"] == food_name].iloc[0]

    # -------- OPTION 4 (PRECISE FULFILLMENT) --------
    if option.startswith("4"):
        remaining = st.session_state.gap - st.session_state.total_consumed

        if st.session_state.option4_done:
            st.success("🎉 You have fulfilled your calorie requirement.")
        else:
            st.write(f"Remaining gap: {remaining:.1f} kcal")

            energy_per_item = row["Serving_Size"] * row["Energy_kcal_g"]
            qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)

            if st.button("Add Snack"):
                max_items_needed = remaining / energy_per_item
                actual_qty = min(qty, max_items_needed)
                energy = actual_qty * energy_per_item

                st.session_state.total_consumed += energy
                st.session_state.option4_list.append((food_name, actual_qty, energy))

                if actual_qty < qty or abs(remaining - energy) < 1e-6:
                    st.session_state.option4_done = True

        for f, q, e in st.session_state.option4_list:
            st.write(f"{f}: {q:.2f} items → {e:.1f} kcal")

    # -------- OPTION 5 (UNCHANGED) --------
    elif option.startswith("5"):
        st.write("Add snacks freely (energy may exceed requirement)")

        qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        if st.button("Add Snack"):
            energy = qty * row["Serving_Size"] * row["Energy_kcal_g"]
            st.session_state.option5_list.append((food_name, qty, energy))

        total_energy = 0
        for f, q, e in st.session_state.option5_list:
            total_energy += e
            st.write(f"{f}: {q} items → {e:.1f} kcal")

        gap_after = st.session_state.gap - total_energy
        if gap_after >= 0:
            st.success(f"Remaining calorie gap: {gap_after:.1f} kcal")
        else:
            st.warning(f"Excess calorie intake: {gap_after:.1f} kcal")
