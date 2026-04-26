import streamlit as st
import helper
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.title('Hybrid Duplicate Question Detection')

q1 = st.text_input('Question 1')
q2 = st.text_input('Question 2')

if st.button('Check Duplicate'):
    if not q1 or not q2:
        st.warning("Please enter both questions.")
    else:
        with st.spinner("Analyzing questions through hybrid pipeline..."):
            decision, metrics = helper.check_duplicate(q1, q2)

        st.subheader("Final Decision")
        if "Duplicate" in decision and "Not" not in decision:
            st.success(decision)
        elif "Not Duplicate" in decision:
            st.error(decision)
        else:
            st.warning(decision)
            
        st.subheader("Pipeline Metrics")
        for key, value in metrics.items():
            if isinstance(value, float):
                st.write(f"**{key}:** {value:.4f}")
            else:
                st.write(f"**{key}:** {value}")