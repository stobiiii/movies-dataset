import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="BB84 Protokoll", layout="wide")
st.title("🔐 BB84 Protokoll – Interaktive Demonstration")

showDetails = st.sidebar.checkbox("Detaillierte Auswertung anzeigen", value=False)

# --- Parameter
st.subheader("⚙️Parameter")
num_bits = st.slider("🔢Anzahl der Bits welche übertragen werden sollen", min_value=10, max_value=10000, value=10000)

#eve_active = st.toggle("🕵️‍♀️Eve aktivieren")


p = st.slider("🕵️‍♀️Agressivität von Eve (Wahrscheinlichkeit, dass Eve ein Photon misst )", 0.0, 100.0, 0.0,format="%.1f%%")
p /=100
eve_active = (p != 0)


q_bob_error = st.slider("🌀Kanalrauschen (Wahrscheinlichkeit, dass Bob ein zufälliges Bit erhält)", 0.0, 100.0, 0.0, format="%.1f%%")
q_bob_error /=100

# --- Alice sendet
bits_alice = np.random.randint(0, 2, num_bits)
basis_alice = np.random.choice(["+", "x"], num_bits)

# --- Eve
eve_measured = np.random.rand(num_bits) < p
basis_eve = np.array(["-" for _ in range(num_bits)], dtype=object)
bits_eve = np.array(["-" for _ in range(num_bits)], dtype=object)

for i in range(num_bits):
    if eve_measured[i]:
        basis_eve[i] = np.random.choice(["+", "x"])
        if basis_eve[i] == basis_alice[i]:
            bits_eve[i] = bits_alice[i]
        else:
            bits_eve[i] = np.random.randint(0, 2)

# --- Bob misst
basis_bob = np.random.choice(["+", "x"], num_bits)
bits_bob = np.empty(num_bits, dtype=int)

for i in range(num_bits):
    if eve_measured[i]:
        send_basis = basis_eve[i]
        send_bit = int(bits_eve[i])
    else:
        send_basis = basis_alice[i]
        send_bit = bits_alice[i]

    if np.random.rand() < q_bob_error:
        bits_bob[i] = np.random.randint(0, 2)
    else:
        if send_basis == basis_bob[i]:
            bits_bob[i] = send_bit
        else:
            bits_bob[i] = np.random.randint(0, 2)

# --- DataFrame
df = pd.DataFrame({
    "Alice Basis": basis_alice,
    "Alice Bit": bits_alice,
    "Eve gemessen?": eve_measured,
    "Eve Basis": basis_eve,
    "Eve Bit": bits_eve,
    "Bob Basis": basis_bob,
    "Bob Bit": bits_bob,
})

#with st.expander("📄Übertragungsverlauf anzeigen"):
st.subheader("📋 Übertragungsverlauf")
st.dataframe(df)


# --- Fehleranalyse
matching_indices = basis_alice == basis_bob
key_alice = bits_alice[matching_indices]
key_bob = bits_bob[matching_indices]

errors = np.sum(key_alice != key_bob)
total_matching = len(key_alice)
error_rate = errors / total_matching if total_matching > 0 else 0


if showDetails:
    st.subheader("📊 Fehleranalyse (Alice → Bob)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gemeinsame Basen (Alice = Bob)", total_matching)
        st.metric("Fehlerhafte Bits", errors)
    with col2:
        st.metric("Fehlerrate", f"{error_rate:.2%}")
else: 
    st.subheader("📊Auswertung")
    st.metric("👩‍💻 ➡️ 👨‍💻 Übertragungsfehler", f"{error_rate:.2%}")

# --- Informationsgewinn Eve

eve_bits_array = bits_eve
bob_bits_array = bits_bob

cond_alice_bob = basis_alice == basis_bob
cond_eve_correct_basis = (basis_eve == basis_alice) & eve_measured & cond_alice_bob
eve_correct_bits = cond_eve_correct_basis & (eve_bits_array == bits_alice)

eve_sure_bits = np.sum(eve_correct_bits)
eve_info_rate = eve_sure_bits / total_matching if total_matching > 0 else 0

if showDetails:
    st.subheader("🕵️‍♀️ Eves Informationsgewinn (sicher erkannt)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sicher abgefangene Schlüsselbits", eve_sure_bits)
    with col2:
        st.metric(" Informationsrate", f"{eve_info_rate:.2%}")
else: 
    st.metric("🕵️‍♀️ Informationsrate", f"{eve_info_rate:.2%}")

# --- Kreisdiagramm
if eve_active:
    cat1 = np.sum(cond_eve_correct_basis & (bob_bits_array == bits_alice))
    cond_eve_wrong_basis = eve_measured & (basis_eve != basis_alice) & cond_alice_bob
    cat2 = np.sum(cond_eve_wrong_basis & (bob_bits_array == bits_alice))
    cat3 = np.sum(cond_eve_wrong_basis & (bob_bits_array != bits_alice))
    cond_eve_none = cond_alice_bob & (~eve_measured)
    cat4 = np.sum(cond_eve_none)
    
    labels = [
        "Eve hat in gleicher Basis gemessen und unerkannt gestohlen",
        "Eve misst in anderer Basis, Bob merkt nichts",
        "Eve misst in anderer Basis, Bob bemerkt Fehler",
        "Photon wurde nicht von Eve gemessen"
    ]
    values = [cat1, cat2, cat3, cat4]
    colors = ["#4CAF50", "#8BC34A", "#F44336", "#2196F3"]
    
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

# --- Sicherheit
def binary_entropy(q):
    if q == 0 or q == 1:
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)

I_AB = 1 - binary_entropy(error_rate)
condition1 = I_AB > eve_info_rate
condition2 = I_AB > 0.5

# Sicher extrahierbare Bits wenn I(E) bekannt
if condition1 and total_matching > 0:
    secure_key_bits = total_matching * (I_AB - eve_info_rate)
    secure_key_percent = (secure_key_bits / num_bits) * 100
else:
    secure_key_bits = 0
    secure_key_percent = 0

# Sicher extrahierbare Bits im Worst Case (I(E) = 0.5)
if condition2 and total_matching > 0:
    secure_key_bits_worst = total_matching * (I_AB - 0.5)
    secure_key_bits_worst = max(secure_key_bits_worst, 0)  # nie negativ
    secure_key_percent_worst = (secure_key_bits_worst / num_bits) * 100
else:
    secure_key_bits_worst = 0
    secure_key_percent_worst = 0


if showDetails:
    st.subheader("🔒 Sicherheit des Schlüssels")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("I(A : B)", f"{I_AB:.4f}")
        st.metric("I(E)", f"{eve_info_rate:.4f}")
    with col2:
        st.metric("Bedingung I(A:B) > I(E)", "✅" if condition1 else "❌")
        st.metric("Worst Case: I(A:B) > 0.5", "✅" if condition2 else "❌")
    
    st.metric("Sicher extrahierbare Bits (bekannte I(E))", f"{secure_key_bits:.1f}")
    st.metric("Prozentualer Anteil am Original (bekannte I(E))", f"{secure_key_percent:.2f}%")
    st.metric("Sicher extrahierbare Bits (Worst Case I(E)=0.5)", f"{secure_key_bits_worst:.1f}")
    st.metric("Prozentualer Anteil am Original (Worst Case)", f"{secure_key_percent_worst:.2f}%")
    
    if total_matching == 0:
        st.warning("Keine gemeinsamen Basen – keine Auswertung möglich.")
    elif condition1:
        st.success("✅ Ein sicherer Schlüssel kann extrahiert werden (I(A:B) > I(E)).")
    elif not condition2:
        st.error("❌ Auch im Worst Case ist keine sichere Extraktion möglich (I(A:B) ≤ 0.5).")
    else:
        st.warning("⚠️ I(A:B) > 0.5, aber ≤ I(E): keine Sicherheit gegenüber aktivem Abhören.")

else: 
    #st.write("✅" if condition1 else "❌")
    #st.metric("Bob hat **mehr** Informationen als Eve" if condition1 else "Eve hat **mehr** Information als Bob", "👨‍💻 > 🕵️‍♀️" if condition1 else "🕵️‍♀️ > 👨‍💻")
    #st.metric("Schlüsselaustausch auch *möglich* wenn Eves Information unbekannt" if condition2 else "Schlüsselaustausch **nicht möglich** wenn Eves Information unbekannt", "✅" if condition2 else "❌")
    if total_matching == 0:
        st.warning("Keine gemeinsamen Basen – keine Auswertung möglich.")
    elif condition2:
        st.success("✅ Ein sicherer Schlüssel kann extrahiert werden auch wenn nicht bekannt ist wie viel Information Eve hat.")
    
    elif condition1:
        st.warning("⚠️ Ein sicherer Schlüssel kann nur extrahiert werden wenn Alice und Bob wissen wie viel Information Eve hat.")
    else:
        st.warning("❌ Ein sicherer Schlüssel kann **nicht** extrahiert werden. Eve hat zu viel Information.")







