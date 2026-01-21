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
for key, val in {
    "predicted": None,
    "gap": None,
    "total_consumed": 0,
    "option4_list": [],
    "option4_done": False,
    "option5_list": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    xls = pd.ExcelFile("Edited.xlsx")
    return (
        pd.read_excel(xls, "Healthy_Adults_Meals"),
        pd.read_excel(xls, "Snack_list_Cor")
    )

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
# VISUAL FUNCTIONS
# ======================
def food_portion_atlas(row, max_items=5):
    items = np.arange(1, max_items + 1)
    energy = items * row["Serving_Size"] * row["Energy_kcal_g"]
    fig, ax = plt.subplots()
    ax.bar(items, energy)
    ax.set_xlabel("Items")
    ax.set_ylabel("Energy (kcal)")
    ax.set_title(f"Portion Atlas: {row['Name']}")
    st.pyplot(fig)

def macro_pie_chart(row):
    fig, ax = plt.subplots()
    ax.pie(
        [row["Carb_g"], row["Protein_g"], row["Fat_g"]],
        labels=["Carbs", "Protein", "Fat"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig)

# ======================
# UI INPUTS
# ======================
st.title("🍎 Personalized Snack Portion Planner")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 64, 25)
height = st.number_input("Height (m)", 1.0, 2.5, 1.65)
weight = st.number_input("Weight (kg)", 30.0, 150.0, 60.0)
daily_intake = st.number_input("Daily Intake (kcal)", 1000, 4000, 1800)
income = st.selectbox("Income Level", [1, 2, 3])

# ======================
# PREDICTION
# ======================
if st.button("Predict Requirement"):
    bmi = weight / height**2
    if not 18.5 <= bmi <= 22.9:
        st.error("Valid only for healthy adults (BMI 18.5–22.9)")
        st.session_state.predicted = None
    else:
        user = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Height (m)": height,
            "Weight (kg)": weight,
            "Daily_total_calory_intake_kcal": daily_intake,
            "Income Level": income
        }])
        st.session_state.predicted = model.predict(user)[0]
        st.session_state.gap = st.session_state.predicted - daily_intake
        st.session_state.total_consumed = 0
        st.session_state.option4_list = []
        st.session_state.option4_done = False
        st.session_state.option5_list = []

# ======================
# DISPLAY
# ======================
if st.session_state.predicted:
    st.success(f"Predicted Requirement: {st.session_state.predicted:.1f} kcal")
    st.info(f"Calorie Gap: {st.session_state.gap:.1f} kcal")

    option = st.radio("Choose Option", [
        "1 - Portion size for given snack",
        "2 - Suitability of eating X items",
        "3 - Fixed snack + suggested snack",
        "4 - Multiple snacks until fulfilled",
        "5 - Multiple snacks (free)"
    ])

    food = st.selectbox("Select Snack", df_snacks["Name"].unique())
    row = df_snacks[df_snacks["Name"] == food].iloc[0]
    kcal_per_item = row["Serving_Size"] * row["Energy_kcal_g"]

    # -------- OPTION 1 --------
    if option.startswith("1"):
        required_items = st.session_state.gap / kcal_per_item
        st.write(f"You need **{required_items:.2f} items** of {food}")
        food_portion_atlas(row)
        macro_pie_chart(row)

    # -------- OPTION 2 --------
    elif option.startswith("2"):
        qty = st.number_input("Items eaten", 0.0, 20.0, 1.0)
        energy = qty * kcal_per_item
        remaining = st.session_state.gap - energy

        if remaining >= 0:
            st.success(f"Suitable. Remaining gap: {remaining:.1f} kcal")
        else:
            st.warning(f"Too much. Excess: {-remaining:.1f} kcal")

    # -------- OPTION 3 --------
    elif option.startswith("3"):
        qty = st.number_input("Fixed snack items", 0.0, 10.0, 1.0)
        fixed_energy = qty * kcal_per_item
        remaining = st.session_state.gap - fixed_energy

        if remaining <= 0:
            st.success("Calorie requirement fulfilled")
        else:
            df_snacks["items_needed"] = remaining / (
                df_snacks["Serving_Size"] * df_snacks["Energy_kcal_g"]
            )
            suggestion = df_snacks.loc[df_snacks["items_needed"].idxmin()]
            st.info(
                f"Suggested: {suggestion['Name']} "
                f"({suggestion['items_needed']:.2f} items)"
            )

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
