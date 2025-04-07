import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ---------------------------
# PAGE CONFIGURATION & TITLE
st.set_page_config(page_title="Music Data Analysis Dashboard", layout="wide")
st.title("🎵 Music Data Analysis Dashboard")

st.markdown("""
## Welcome to Our Music Data Analsysis Presentation for Data fluency !

### Research Question
**How do user demographics and music streaming behaviors influence musical preferences and mental health outcomes?**

### Our Hypotheses
- **H1:** Higher listening hours are correlated with higher anxiety scores.
- **H2:** Younger users tend to prefer more upbeat genres (e.g., EDM, Hip hop, K pop), while older users lean toward genres like Classical and Rock.
- **H3:** Users who engage in music creation (e.g., instrumentalists, composers) exhibit distinct listening patterns and mental health outcomes compared to those who do not.

### Methodology
- **Data Overview:**  
  We work with a clean, well-prepared dataset capturing listening habits, genre preferences, and self-reported mental health indicators.
  
- **Analytical Approach:**  
  1. **Exploratory Data Analysis (EDA):** Use interactive visualizations (histograms, bar charts, box plots, and scatter plots) to reveal trends and patterns.
  2. **Hypothesis Testing:** Employ scatter plots with trendlines and compute Pearson correlations to validate our hypotheses.
  3. **Key Performance Indicators (KPIs):** Derive metrics like average listening hours, genre diversity scores, and the distribution of music effects to inform strategic recommendations.

### Dashboard Structure
- **Overview:** Dataset snapshot and descriptive statistics.
- **Exploratory Analysis:** In-depth visualizations that uncover the data story.
- **Hypothesis Testing:** Statistical validation of our research hypotheses.
- **KPIs & Recommendations:** Actionable insights and strategic next steps.

Let's dive in and discover the story behind the music data!
""")

# ---------------------------
# SIDEBAR FOR NAVIGATION
# ---------------------------
section = st.sidebar.selectbox("Select Analysis Section", 
                               ["Overview", "Exploratory Data Analysis", "Hypothesis Testing", "KPIs & Recommendations"])

# ---------------------------
# DATA LOADING FUNCTION
# ---------------------------
@st.cache(allow_output_mutation=True)
def load_data():
    # Load the CSV file (ensure music_data.csv is in the same folder)
    data = pd.read_csv("music_data.csv")
    
    # Convert Timestamp and Date columns to datetime objects
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    
    # Convert numeric columns
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data['Hours per day'] = pd.to_numeric(data['Hours per day'], errors='coerce')
    data['BPM'] = pd.to_numeric(data['BPM'], errors='coerce')
    
    # Map frequency categorical variables (assumed to be: "Never", "Rarely", "Sometimes", "Very frequently")
    freq_order = {"Never":0, "Rarely":1, "Sometimes":2, "Very frequently":3}
    freq_cols = [col for col in data.columns if "Frequency" in col]
    for col in freq_cols:
        data[col + "_num"] = data[col].map(freq_order)
    
    # Ensure mental health score columns are numeric
    mental_cols = ["Anxiety", "Depression", "Insomnia", "OCD"]
    for col in mental_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

data = load_data()

# ---------------------------
# SECTION: OVERVIEW
# ---------------------------
if section == "Overview":
    st.header("1. Overview")
    st.markdown("""
    **Our Data at a Glance**

    Our dataset captures the music streaming habits of users—ranging from their age and listening hours to their favorite genres—and how these behaviors relate to self‐reported mental health scores.
    
    Our goal is to understand the interplay between **music consumption** and **mental health**.
    """)
    
    # Compute key metrics
    total_records = len(data)
    avg_age = data['Age'].mean()
    avg_hours = data['Hours per day'].mean()
    unique_services = data["Primary streaming service"].nunique()

    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", total_records)
    col2.metric("Average Age", f"{avg_age:.1f} years")
    col3.metric("Avg. Listening Hours", f"{avg_hours:.1f} hrs/day")
    col4.metric("Streaming Services", unique_services)
    
    st.markdown("### Quick Story")
    st.markdown(f"""
    - **Diverse Audience:** Our users span multiple age groups.
    - **Engagement:** On average, users spend about **{avg_hours:.1f} hours/day** listening to music.
    - **Varied Choices:** Multiple streaming services and genres are represented, providing a rich canvas to explore how music impacts mental health.
    """)


# ---------------------------
# SECTION: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------
elif section == "Exploratory Data Analysis":
    st.header("3. Exploratory Data Analysis (EDA)")
    st.markdown("""
    In this section we explore how musical preferences and listening behaviors relate to mental health.
    Our visualizations use a consistent color palette and are arranged side‐by‐side for easy comparison.
    """)

    # --- Mental Health Scores Overview ---
    st.subheader("Mental Health Scores Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig_anxiety = px.box(data, y="Anxiety", title="Anxiety Scores",
                             color_discrete_sequence=px.colors.qualitative.Set1)
        fig_anxiety.update_layout(template="simple_white")
        st.plotly_chart(fig_anxiety, use_container_width=True)
    with col2:
        fig_depression = px.box(data, y="Depression", title="Depression Scores",
                             color_discrete_sequence=px.colors.qualitative.Set1)
        fig_depression.update_layout(template="simple_white")
        st.plotly_chart(fig_depression, use_container_width=True)
    
    st.subheader("Other Mental Health Metrics")
    col1, col2 = st.columns(2)
    with col1:
        fig_insomnia = px.box(data, y="Insomnia", title="Insomnia Scores",
                             color_discrete_sequence=px.colors.qualitative.Set1)
        fig_insomnia.update_layout(template="simple_white")
        st.plotly_chart(fig_insomnia, use_container_width=True)
    with col2:
        fig_ocd = px.box(data, y="OCD", title="OCD Scores",
                             color_discrete_sequence=px.colors.qualitative.Set1)
        fig_ocd.update_layout(template="simple_white")
        st.plotly_chart(fig_ocd, use_container_width=True)
    
    # --- Correlation Focused on Music and Mental Health ---
    st.subheader("Correlation: Listening Hours & Mental Health")
    relevant_cols = ["Hours per day", "Anxiety", "Depression", "Insomnia", "OCD"]
    corr = data[relevant_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True,
                         title="Correlation Matrix: Listening & Mental Health",
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_corr.update_layout(template="simple_white")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # --- Streaming Service & Mental Health ---
    st.subheader("Streaming Service & Anxiety")
    col1, col2 = st.columns(2)
    with col1:
        # Re-use our streaming service counts created earlier
        fig_service = px.bar(service_counts, x='Primary streaming service', y='count',
                             title="Streaming Service Distribution",
                             labels={"Primary streaming service": "Streaming Service", "count": "Count"},
                             color='Primary streaming service',
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig_service.update_layout(xaxis={'categoryorder':'total descending'}, template="simple_white")
        st.plotly_chart(fig_service, use_container_width=True)
    with col2:
        fig_service_box = px.box(data, x="Primary streaming service", y="Anxiety", 
                                  title="Anxiety by Streaming Service", 
                                  color="Primary streaming service",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
        fig_service_box.update_layout(template="simple_white")
        st.plotly_chart(fig_service_box, use_container_width=True)
    st.header("3. Exploratory Data Analysis (EDA)")
    st.markdown("This section uses interactive visualizations to uncover patterns and trends. The charts are designed using best practices for clarity and balance.")

    # --- Demographic Analysis ---
    st.subheader("Demographic Analysis")
    fig_age = px.histogram(data, x="Age", nbins=20, title="Age Distribution",
                           color_discrete_sequence=px.colors.qualitative.Set1)
    fig_age.update_layout(title_font=dict(size=20, color="black"), template="simple_white")
    st.plotly_chart(fig_age, use_container_width=True)
    
    # --- Streaming Service Usage ---
    st.subheader("Streaming Service Usage")
    # Create a value_counts dataframe and reset index, naming the count column explicitly
    service_counts = data["Primary streaming service"].value_counts().reset_index(name='count')
    # Now the columns are: "index" (which holds the service name) and "count"
    # Alternatively, you can rename "index" to "Primary streaming service"
    service_counts = service_counts.rename(columns={"index": "Primary streaming service"})
    fig_service = px.bar(service_counts, x='Primary streaming service', y='count',
                         title="Distribution of Streaming Services",
                         labels={"Primary streaming service": "Streaming Service", "count": "Count"},
                         color='Primary streaming service',
                         color_discrete_sequence=px.colors.qualitative.Set2)
    fig_service.update_layout(xaxis={'categoryorder':'total descending'}, template="simple_white")
    st.plotly_chart(fig_service, use_container_width=True)
    
    # --- Listening Hours ---
    st.subheader("Listening Hours per Day")
    fig_hours = px.histogram(data, x="Hours per day", nbins=15, title="Distribution of Listening Hours",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_hours.update_layout(template="simple_white")
    st.plotly_chart(fig_hours, use_container_width=True)
    
    # --- Genre Preferences ---
    st.subheader("Favorite Genre Distribution")
    # Create a dataframe with proper column names using rename_axis and reset_index
    genre_counts = data["Fav genre"].value_counts().rename_axis('Genre').reset_index(name='Count')
    fig_genre = px.bar(genre_counts, x='Genre', y='Count',
                    title="Favorite Genre Distribution",
                    labels={"Genre": "Genre", "Count": "Count"},
                    color='Genre', color_discrete_sequence=px.colors.qualitative.Bold)
    fig_genre.update_layout(xaxis={'categoryorder':'total descending'}, template="simple_white")
    st.plotly_chart(fig_genre, use_container_width=True)
    
    # --- Mental Health Scores ---
   # --- Mental Health Scores ---
    st.subheader("Mental Health Scores")
    mental_cols = ["Anxiety", "Depression", "Insomnia", "OCD"]
    for col in mental_cols:
        fig = px.box(data, y=col, title=f"Distribution of {col} Scores",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Correlation Matrix ---
    st.subheader("Correlation Matrix")
    numeric_cols = ["Age", "Hours per day", "BPM"] + mental_cols
    corr = data[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix",
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_corr.update_layout(template="simple_white")
    st.plotly_chart(fig_corr, use_container_width=True)
# ---------------------------
# SECTION: HYPOTHESIS TESTING
# ---------------------------
elif section == "Hypothesis Testing":
    st.header("4. Hypothesis Testing & Advanced Analysis")
    st.markdown("""
    **Example Hypothesis:**  
    _Higher listening hours are correlated with higher anxiety scores._  
    
    We explore this hypothesis using scatter plots with trend lines and calculate the Pearson correlation.
    """)
    
    # Scatter Plot with Trend Line
    fig_scatter = px.scatter(data, x="Hours per day", y="Anxiety", trendline="ols",
                             title="Hours per Day vs Anxiety", color_discrete_sequence=px.colors.qualitative.Set1)
    fig_scatter.update_layout(template="simple_white")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Pearson Correlation Calculation
    valid_data = data[["Hours per day", "Anxiety"]].dropna()
    corr_value, p_value = stats.pearsonr(valid_data["Hours per day"], valid_data["Anxiety"])
    st.write(f"**Pearson Correlation Coefficient:** {corr_value:.2f} (p-value: {p_value:.3f})")
    
    st.markdown("#### Comparison of Anxiety Scores by Streaming Service")
    box_fig = px.box(data, x="Primary streaming service", y="Anxiety", 
                     title="Anxiety Scores by Streaming Service", 
                     color="Primary streaming service", color_discrete_sequence=px.colors.qualitative.Set2)
    box_fig.update_layout(template="simple_white")
    st.plotly_chart(box_fig, use_container_width=True)
    
    st.markdown("Additional statistical tests (e.g., t-tests, ANOVAs) can be performed to compare subgroups and further validate these relationships.")

# ---------------------------
# SECTION: KPIs & RECOMMENDATIONS
# ---------------------------
elif section == "KPIs & Recommendations":
    st.header("5. Key Performance Indicators (KPIs) & Recommendations")
    st.markdown("### Summary of Key Insights")
    
    # Calculate average listening hours
    avg_hours = round(data["Hours per day"].mean(), 2)
    st.markdown(f"- **Average Listening Hours per Day:** {avg_hours} hours")
    
    # Genre Diversity Score (as an example: count number of genres each user listens to frequently)
    freq_cols_num = [col for col in data.columns if col.endswith("_num") and "Frequency" in col]
    data['Genre_Diversity'] = data[freq_cols_num].ge(2).sum(axis=1)
    avg_genre_diversity = round(data['Genre_Diversity'].mean(), 2)
    st.markdown(f"- **Average Genre Diversity Score:** {avg_genre_diversity} (on an ordinal scale)")
    
    # Music Effects Ratio (if the column exists)
    if "Music effects" in data.columns:
        effect_counts = data["Music effects"].value_counts()
        total_effect = effect_counts.sum()
        ratio_improve = effect_counts.get("Improve", 0) / total_effect if total_effect > 0 else 0
        ratio_worsen = effect_counts.get("Worsen", 0) / total_effect if total_effect > 0 else 0
        st.markdown("#### Music Effects Distribution:")
        st.write(effect_counts)
        st.write(f"**Improve Ratio:** {ratio_improve:.2f}")
        st.write(f"**Worsen Ratio:** {ratio_worsen:.2f}")
    
    st.markdown("### Recommendations")
    st.markdown("""
    - **Tailored Playlists:**  
      Design playlists and recommendations that align with users’ genre diversity and listening habits. For example, younger users preferring upbeat genres (e.g., EDM, Hip hop) could be offered dynamic playlists that may positively impact their mood.
    
    - **User Engagement Strategies:**  
      Focus on strategies that increase listening hours while promoting genres associated with positive mental health outcomes.
    
    - **Further Analysis:**  
      Perform A/B testing to determine the causal relationship between listening behavior and self-reported mental health outcomes. Incorporate user feedback to refine models and dashboard features.
    
    - **Design & Storytelling:**  
      Use clear, uncluttered visualizations (with a consistent color palette and strong titles) to effectively communicate insights to stakeholders.
    """)
    
    st.markdown("### Next Steps")
    st.markdown("""
    - Iteratively refine your data cleaning and transformation processes.
    - Develop interactive dashboards for real-time insights.
    - Engage with stakeholders using the data storytelling approach to drive actionable decisions.
    """)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("""
**Data Analysis Dashboard** built using Streamlit By group 3.
""")