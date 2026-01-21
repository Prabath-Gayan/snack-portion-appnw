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

# Rename snack columns
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
# VISUALIZATION FUNCTIONS
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
    ax.pie(macros.values(), labels=macros.keys(), autopct="%1.1f%%",
           startangle=90, colors=["#FFD700", "#87CEFA", "#FF6347"])
    ax.set_title(f"Macronutrient Distribution: {row['Name']}")
    ax.axis("equal")
    st.pyplot(fig)


# ======================
# UI
# ======================
st.title("🍎 Personalized Snack Portion Planner")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 64, 25)
height = st.number_input("Height (m)", 1.0, 2.5, 1.65)
weight = st.number_input("Weight (kg)", 30.0, 150.0, 60.0)
daily_intake = st.number_input("Daily Calorie Intake (kcal)", 1000, 4000, 1800)
income = st.selectbox("Income Level", [1, 2, 3])

# ======================
# STEP 1: PREDICT
# ======================
if st.button("Predict Requirement"):
    bmi = weight / (height ** 2)

    if bmi < 18.5 or bmi > 22.9:
        st.error("This model is intended for healthy adults (BMI 18.5–22.9).")
        st.session_state.predicted = None
        st.session_state.gap = None
        st.session_state.total_consumed = 0
        st.session_state.option4_list = []
        st.session_state.option5_list = []
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
        gap = pred - daily_intake

        st.session_state.predicted = pred
        st.session_state.gap = gap
        st.session_state.total_consumed = 0
        st.session_state.option4_list = []
        st.session_state.option5_list = []

# ======================
# DISPLAY PREDICTION
# ======================
if st.session_state.predicted is not None:
    remaining_gap = st.session_state.gap - st.session_state.total_consumed
    st.success(f"Predicted Requirement: {st.session_state.predicted:.1f} kcal/day")
    st.info(f"Calorie Gap: {remaining_gap:.1f} kcal")

    if remaining_gap <= 0:
        st.info("No additional snack needed.")

# ======================
# SNACK OPTIONS MENU
# ======================
if st.session_state.predicted is not None and remaining_gap > 0:
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

    # -------- COMMON INPUT --------
    food_name = st.selectbox("Select Snack", df_snacks["Name"].unique())
    row = df_snacks[df_snacks["Name"] == food_name].iloc[0]

    # -----------------
    # OPTION 1, 2, 3 (unchanged)
    # -----------------
    if option.startswith("1"):
        st.write(f"Remaining gap: {remaining_gap:.1f} kcal")
        portion = remaining_gap / row["Energy_kcal_g"]
        items = portion / row["Serving_Size"]

        st.write(f"Required portion: {portion:.1f} {row['Size_unit']}")
        st.write(f"Required items : {items:.1f}")

        food_portion_atlas(row)
        macro_pie_chart(row)

    elif option.startswith("2"):
        items_input = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        energy = items_input * row["Serving_Size"] * row["Energy_kcal_g"]
        diff = remaining_gap - energy

        st.write(f"Energy consumed: {energy:.1f} kcal")
        st.write(f"{'Remaining gap' if diff>=0 else 'Excess intake'}: {abs(diff):.1f} kcal")

        food_portion_atlas(row)
        macro_pie_chart(row)

    elif option.startswith("3"):
        items_input1 = st.number_input("Items eaten for first snack", 0.0, 50.0, 1.0)
        energy1 = items_input1 * row["Serving_Size"] * row["Energy_kcal_g"]
        remaining_after1 = remaining_gap - energy1

        st.write(f"{food_name}: {energy1:.1f} kcal")
        food_portion_atlas(row)
        macro_pie_chart(row)

        if remaining_after1 > 0:
            food_name2 = st.selectbox("Select second snack", df_snacks["Name"].unique(), key="second_snack")
            row2 = df_snacks[df_snacks["Name"] == food_name2].iloc[0]
            portion2 = remaining_after1 / row2["Energy_kcal_g"]
            items2 = portion2 / row2["Serving_Size"]

            st.write(f"{food_name2} portion: {portion2:.1f} {row2['Size_unit']} ({items2:.1f} items)")
            food_portion_atlas(row2)
            macro_pie_chart(row2)
            
       # -------- OPTION 4 --------
    elif option.startswith("4"):
        remaining = st.session_state.gap - st.session_state.total_consumed

        if st.session_state.option4_done:
            st.success("🎉 Requirement fulfilled")
        else:
            qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)
            if st.button("Add Snack"):
                max_items = remaining / kcal_per_item
                actual = min(qty, max_items)
                energy = actual * kcal_per_item
                st.session_state.total_consumed += energy
                st.session_state.option4_list.append((food, actual, energy))
                if actual < qty or abs(remaining - energy) < 1e-6:
                    st.session_state.option4_done = True

        for f, q, e in st.session_state.option4_list:
            st.write(f"{f}: {q:.2f} items → {e:.1f} kcal")

    # -------- OPTION 5 --------
    elif option.startswith("5"):
        qty = st.number_input("Items eaten", 0.0, 50.0, 1.0)
        if st.button("Add Snack"):
            energy = qty * kcal_per_item
            st.session_state.option5_list.append((food, qty, energy))

        total = sum(e for _, _, e in st.session_state.option5_list)
        for f, q, e in st.session_state.option5_list:
            st.write(f"{f}: {q} → {e:.1f} kcal")

        gap_after = st.session_state.gap - total
        st.info(f"Gap after snacks: {gap_after:.1f} kcal")
