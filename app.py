"""Stakeholder-facing summary of the customer outreach ranking results."""

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Customer outreach prioritization", page_icon="🛍️", layout="wide")
st.title("Customer outreach prioritization")
st.caption("A simple view of how purchase history could help focus a limited customer contact budget.")

st.subheader("Who should the company contact first?")
st.write(
    "If the company can contact only 1 in 5 active customers, this model ranks customers by their recent purchase history. "
    "In the November 2011 historical test, the highest-ranked group purchased more often than customers overall."
)

top_col, all_col, count_col = st.columns(3)
top_col.metric("Purchase rate in highest-ranked 20%", "74.8%")
all_col.metric("Purchase rate across all active customers", "42.9%")
count_col.metric("Customers in the shortlist", "493 of 2,461")

st.markdown("#### What this could mean for the business")
st.write(
    "About **369 of the 493** highest-ranked customers purchased within 30 days in this historical test. "
    "A team could use a ranking like this to focus limited outreach capacity on a smaller, higher-potential group."
)

rates = pd.DataFrame(
    {
        "Customer group": ["All customers", "Top 20%"],
        "Purchase rate": [42.9, 74.8],
    }
)
bars = alt.Chart(rates).mark_bar(size=34, color="#2C7FB8").encode(
    x=alt.X(
        "Purchase rate:Q",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(title="Purchased within 30 days (%)", values=[0, 20, 40, 60, 80, 100]),
    ),
    y=alt.Y("Customer group:N", sort=None, axis=alt.Axis(title=None)),
    tooltip=[alt.Tooltip("Customer group:N"), alt.Tooltip("Purchase rate:Q", format=".1f")],
)
labels = bars.mark_text(align="left", baseline="middle", dx=6, color="#E6EAF0").encode(
    text=alt.Text("Purchase rate:Q", format=".1f")
)
st.altair_chart((bars + labels).properties(height=180), use_container_width=True)
st.caption("The top 20% are the customers with the highest model scores. Each bar shows the share who purchased within 30 days.")

st.info(
    "**Recommended next step:** test the shortlist in a controlled campaign. Keep a comparable group uncontacted, "
    "then compare purchases, revenue, and profit. This historical result shows who was likely to buy; it does not show that outreach caused the purchases."
)

with st.expander("How the model made this shortlist"):
    st.markdown(
        "The model used each customer's previous 90 days of purchase activity to rank the chance of a purchase in the next 30 days. "
        "The shortlist contains the 20% with the highest scores."
    )
    st.markdown("**Ranking quality:** average precision was **0.675**. At 20% contact capacity, the model reached **34.9%** of November purchasers.")
    st.markdown("**Main signals:** number of orders, total spend, days since last purchase, and quantity bought. These are patterns in the data, not proof of cause and effect.")
    st.caption(
        "The scores are useful for ordering customers, not as exact purchase probabilities. "
        "This result comes from one UK retailer and one historical test month, so performance may differ in another period or business."
    )
