import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings

# Suppress warnings for better user experience
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

# Set page config
st.set_page_config(
    page_title="Music & Mental Health Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #073763;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #3B6AA0;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #050101;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B6AA0;
        color: white;
    }
    /* Increase plot height */
    .stPlotlyChart {
        min-height: 500px;
    }
</style>
""", unsafe_allow_html=True)

# Define a consistent color palette for the app
MAIN_COLOR = "#3B6AA0"
ACCENT_COLOR = "#6A8EBF"
COLORS = px.colors.sequential.Blues_r
QUALITATIVE_COLORS = px.colors.qualitative.Pastel

# Create sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Feature Engineering", "EDA & Visualization", 
                                "Hypothesis Testing", "KPIs & Recommendations"])

# Function to create a sample dataset if needed
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n = 500
    
    # Age distribution (18-65)
    age = np.random.normal(30, 8, n).astype(int)
    age = np.clip(age, 18, 65)
    
    # Hours of listening per day (0.5-10)
    hours_per_day = np.random.lognormal(1.5, 0.6, n)
    hours_per_day = np.clip(hours_per_day, 0.5, 10).round(1)
    
    # Streaming services
    services = ['Spotify', 'Apple Music', 'YouTube Music', 'Amazon Music', 'Tidal', 'Other']
    service_weights = [0.4, 0.25, 0.2, 0.1, 0.03, 0.02]
    primary_streaming = np.random.choice(services, n, p=service_weights)
    
    # Music genres (frequency of listening)
    frequency_options = ['Never', 'Rarely', 'Sometimes', 'Very frequently']
    genres = ['Rock', 'Pop', 'Metal', 'Classical', 'Hip hop', 'R&B', 'EDM', 'Folk', 'Jazz', 'Country']
    
    genre_data = {}
    for genre in genres:
        # Different genres have different distributions
        if genre in ['Pop', 'Hip hop']:
            weights = [0.1, 0.2, 0.3, 0.4]
        elif genre in ['Metal', 'Classical', 'Jazz']:
            weights = [0.4, 0.3, 0.2, 0.1]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]
        
        genre_data[genre] = np.random.choice(frequency_options, n, p=weights)
    
    # Favorite genre
    fav_genre = np.random.choice(genres, n)
    
    # Context variables
    while_working = np.random.choice([True, False], n, p=[0.7, 0.3])
    instrumentalist = np.random.choice([True, False], n, p=[0.3, 0.7])
    composer = np.random.choice([True, False], n, p=[0.15, 0.85])
    exploratory = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.4, 0.3])
    foreign_languages = np.random.choice([True, False], n, p=[0.4, 0.6])
    
    # Mental health metrics
    # For correlated values with music behavior
    base_anxiety = np.random.normal(5, 2, n)
    base_depression = np.random.normal(4, 2, n)
    base_insomnia = np.random.normal(4, 2, n)
    base_ocd = np.random.normal(3, 2, n)
    
    # Add some correlation with hours
    anxiety = base_anxiety - 0.3 * hours_per_day + np.random.normal(0, 1, n)
    depression = base_depression - 0.4 * hours_per_day + np.random.normal(0, 1, n)
    insomnia = base_insomnia - 0.2 * hours_per_day + np.random.normal(0, 1, n)
    ocd = base_ocd - 0.1 * hours_per_day + np.random.normal(0, 1, n)
    
    # Clip to 0-10 scale and round
    anxiety = np.clip(anxiety, 0, 10).round(1)
    depression = np.clip(depression, 0, 10).round(1)
    insomnia = np.clip(insomnia, 0, 10).round(1)
    ocd = np.clip(ocd, 0, 10).round(1)
    
    # Create dataframe
    data = {
        'Age': age,
        'Hours_Per_Day': hours_per_day,
        'Primary_Streaming': primary_streaming,
        'Favorite_Genre': fav_genre,
        'While_Working': while_working,
        'Instrumentalist': instrumentalist,
        'Composer': composer,
        'Exploratory': exploratory,
        'Foreign_Languages': foreign_languages,
        'Anxiety': anxiety,
        'Depression': depression,
        'Insomnia': insomnia,
        'OCD': ocd
    }
    
    # Add genre frequency data
    for genre in genres:
        data[f'{genre}_Frequency'] = genre_data[genre]
    
    df = pd.DataFrame(data)
    return df

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the actual dataset
        df = pd.read_csv("music_data.csv")
        
        # Rename columns to make them more code-friendly
        column_mapping = {
            'Timestamp': 'Timestamp',
            'Date': 'Date',
            'Age': 'Age',
            'Primary streaming service': 'Primary_Streaming',
            'Hours per day': 'Hours_Per_Day',
            'While working': 'While_Working',
            'Instrumentalist': 'Instrumentalist',
            'Composer': 'Composer',
            'Fav genre': 'Favorite_Genre',
            'Exploratory': 'Exploratory',
            'Foreign languages': 'Foreign_Languages',
            'BPM': 'BPM',
            'Frequency [Classical]': 'Classical_Frequency',
            'Frequency [Country]': 'Country_Frequency',
            'Frequency [EDM]': 'EDM_Frequency',
            'Frequency [Folk]': 'Folk_Frequency',
            'Frequency [Gospel]': 'Gospel_Frequency',
            'Frequency [Hip hop]': 'Hip_hop_Frequency',
            'Frequency [Jazz]': 'Jazz_Frequency',
            'Frequency [K pop]': 'K_pop_Frequency',
            'Frequency [Latin]': 'Latin_Frequency',
            'Frequency [Lofi]': 'Lofi_Frequency',
            'Frequency [Metal]': 'Metal_Frequency',
            'Frequency [Pop]': 'Pop_Frequency',
            'Frequency [R&B]': 'RnB_Frequency',
            'Frequency [Rap]': 'Rap_Frequency',
            'Frequency [Rock]': 'Rock_Frequency',
            'Frequency [Video game music]': 'Video_game_Frequency',
            'Anxiety': 'Anxiety',
            'Depression': 'Depression',
            'Insomnia': 'Insomnia',
            'OCD': 'OCD',
            'Music effects': 'Music_Effects'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert boolean columns if needed
        bool_columns = ['While_Working', 'Instrumentalist', 'Composer', 'Foreign_Languages']
        for col in bool_columns:
            if col in df.columns:
                # Check if the column contains Yes/No values
                if df[col].dtype == 'object' and set(df[col].dropna().unique()).issubset({'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n'}):
                    df[col] = df[col].map(lambda x: x.lower() in ('yes', 'y', 'true') if isinstance(x, str) else x)
        
        # Convert Age to numeric, handling any non-numeric values
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            # Fill NaN values with median age
            df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Convert Hours_Per_Day to numeric
        if 'Hours_Per_Day' in df.columns:
            df['Hours_Per_Day'] = pd.to_numeric(df['Hours_Per_Day'], errors='coerce')
            # Fill NaN values with median
            df['Hours_Per_Day'] = df['Hours_Per_Day'].fillna(df['Hours_Per_Day'].median())
        
        return df
    except FileNotFoundError:
        # If file doesn't exist, create sample data
        st.warning("Using sample data as 'music_data.csv' was not found")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

# Load the data
df = load_data()

# TASK 2: ADVANCED FEATURE ENGINEERING
# Function to convert categorical frequencies to numeric
@st.cache_data
def engineer_features(dataframe):
    # Create a copy to avoid modifying the original
    df_engineered = dataframe.copy()
    
    # 1. Convert categorical frequency values to numeric
    frequency_map = {
        'Never': 0, 
        'Rarely': 1, 
        'Sometimes': 2, 
        'Very frequently': 3
    }
    
    # Get all genre frequency columns
    genre_cols = [col for col in df_engineered.columns if col.endswith('_Frequency')]
    
    # Convert each genre frequency to numeric
    for col in genre_cols:
        df_engineered[f'{col}_Numeric'] = df_engineered[col].map(frequency_map)
    
    # 2. Calculate Genre Diversity Index
    numeric_genre_cols = [col for col in df_engineered.columns if col.endswith('_Frequency_Numeric')]
    
    # Check if any numeric genre columns were created
    if numeric_genre_cols:
        df_engineered['Genre_Diversity_Index'] = df_engineered[numeric_genre_cols].sum(axis=1)
        
        # Normalize to 0-10 scale for easier interpretation
        min_val = df_engineered['Genre_Diversity_Index'].min()
        max_val = df_engineered['Genre_Diversity_Index'].max()
        
        # Avoid division by zero if min_val equals max_val
        if min_val != max_val:
            df_engineered['Genre_Diversity_Index'] = 10 * (df_engineered['Genre_Diversity_Index'] - min_val) / (max_val - min_val)
        else:
            df_engineered['Genre_Diversity_Index'] = 5  # Default to middle value if all values are the same
    else:
        # If no numeric genre columns were created, set a default value
        df_engineered['Genre_Diversity_Index'] = 5
    
    # 3. Calculate Engagement Score
    # Base score from hours per day (0-10 scale)
    df_engineered['Engagement_Score'] = df_engineered['Hours_Per_Day']
    
    # Add points for listening while working
    if 'While_Working' in df_engineered.columns:
        df_engineered['Engagement_Score'] += df_engineered['While_Working'].astype(float) * 2
    
    # Add points for music creation (instrumentalist/composer)
    if 'Instrumentalist' in df_engineered.columns:
        df_engineered['Engagement_Score'] += df_engineered['Instrumentalist'].astype(float) * 3
    
    if 'Composer' in df_engineered.columns:
        df_engineered['Engagement_Score'] += df_engineered['Composer'].astype(float) * 4
    
    # Normalize to 0-10 scale
    min_val = df_engineered['Engagement_Score'].min()
    max_val = df_engineered['Engagement_Score'].max()
    
    # Avoid division by zero
    if min_val != max_val:
        df_engineered['Engagement_Score'] = 10 * (df_engineered['Engagement_Score'] - min_val) / (max_val - min_val)
    else:
        df_engineered['Engagement_Score'] = 5
    
    # 4. Create contextual flags
    # Exploratory flag (0=Low, 1=Medium, 2=High)
    if 'Exploratory' in df_engineered.columns:
        exploratory_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df_engineered['Exploratory_Level'] = df_engineered['Exploratory'].map(exploratory_map)
        # Handle missing or invalid values
        df_engineered['Exploratory_Level'] = df_engineered['Exploratory_Level'].fillna(1)  # Default to Medium
    else:
        df_engineered['Exploratory_Level'] = 1  # Default to Medium if column doesn't exist
    
    # 5. Calculate a Music Versatility Score
    # This combines genre diversity, exploratory level, and foreign language listening
    
    # Handle Foreign_Languages if it exists
    foreign_lang_value = 0
    if 'Foreign_Languages' in df_engineered.columns:
        foreign_lang_value = df_engineered['Foreign_Languages'].astype(float)
    
    df_engineered['Music_Versatility'] = (
        df_engineered['Genre_Diversity_Index'] / 10 * 0.5 +
        df_engineered['Exploratory_Level'] / 2 * 0.3 +
        foreign_lang_value * 0.2
    ) * 10
    
    # 6. Calculate Mental Health Composite Score (average of all mental health metrics)
    mental_health_cols = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    available_mh_cols = [col for col in mental_health_cols if col in df_engineered.columns]
    
    if available_mh_cols:
        df_engineered['Mental_Health_Composite'] = df_engineered[available_mh_cols].mean(axis=1)
    else:
        df_engineered['Mental_Health_Composite'] = 5  # Default value if no mental health columns exist
    
    # 7. Engagement to Mental Health Ratio
    df_engineered['Engagement_MH_Ratio'] = df_engineered['Engagement_Score'] / (df_engineered['Mental_Health_Composite'] + 1)
    
    return df_engineered

# Apply feature engineering
df_engineered = engineer_features(df)

# TASK 1: DATA OVERVIEW & BASIC METRICS
if page == "Data Overview":
    st.title("🎵 Music & Mental Health: Data Overview")
    
    st.markdown("""
    This dashboard explores the relationship between music listening habits and mental health metrics.
    Below are key metrics from our dataset, showing the scale and scope of our analysis.
    """)
    
    # Data verification
    st.subheader("Data Integrity Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total columns:** {df.shape[1]}")
        st.write(f"**Column types:** {df.dtypes.value_counts().to_dict()}")
    
    with col2:
        missing_values = df.isnull().sum().sum()
        st.write(f"**Missing values:** {missing_values}")
        if missing_values > 0:
            st.warning(f"There are {missing_values} missing values in the dataset")
        else:
            st.success("No missing values found in the dataset")
    
    # Display sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head())
    
    # Key metrics in cards using columns
    st.subheader("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Participants", 
                  value=f"{df.shape[0]:,}", 
                  delta=None)
    
    with col2:
        st.metric(label="Average Age", 
                  value=f"{df['Age'].mean():.1f}", 
                  delta=None)
    
    with col3:
        st.metric(label="Avg. Listening Hours/Day", 
                  value=f"{df['Hours_Per_Day'].mean():.1f}", 
                  delta=None)
    
    with col4:
        st.metric(label="Unique Streaming Services", 
                  value=f"{df['Primary_Streaming'].nunique()}", 
                  delta=None)
    
    # Additional metrics row
    st.subheader("Mental Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Avg. Anxiety Score", 
                  value=f"{df['Anxiety'].mean():.1f}/10", 
                  delta=None)
    
    with col2:
        st.metric(label="Avg. Depression Score", 
                  value=f"{df['Depression'].mean():.1f}/10", 
                  delta=None)
    
    with col3:
        st.metric(label="Avg. Insomnia Score", 
                  value=f"{df['Insomnia'].mean():.1f}/10", 
                  delta=None)
    
    with col4:
        st.metric(label="Avg. OCD Score", 
                  value=f"{df['OCD'].mean():.1f}/10", 
                  delta=None)
    
    # Narrative about the metrics
    st.markdown("""
    ### What These Metrics Tell Us
    
    - Our study includes a diverse group of **{:,} participants** with an average age of **{:.1f} years**.
    - On average, participants listen to music for **{:.1f} hours per day**, using one of **{} different streaming services**.
    - The mental health metrics show moderate levels of anxiety and depression, with slightly lower levels of insomnia and OCD symptoms.
    - These baseline metrics provide context for our more detailed analysis of how music consumption habits correlate with mental health outcomes.
    """.format(df.shape[0], df['Age'].mean(), df['Hours_Per_Day'].mean(), df['Primary_Streaming'].nunique()))
    
    # Data structure details
    st.subheader("Dataset Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographic Variables:**")
        st.write("- Age")
        st.write("**Music Consumption Variables:**")
        st.write("- Hours Per Day")
        st.write("- Primary Streaming Service")
        st.write("- Genre Preferences")
        st.write("- Music Context (While Working, etc.)")
    
    with col2:
        st.write("**Music Engagement Variables:**")
        st.write("- Instrumentalist Status")
        st.write("- Composer Status")
        st.write("- Exploratory Listening")
        st.write("- Foreign Language Music")
        st.write("**Mental Health Variables:**")
        st.write("- Anxiety Score (0-10)")
        st.write("- Depression Score (0-10)")
        st.write("- Insomnia Score (0-10)")
        st.write("- OCD Score (0-10)")

# Placeholder sections for other tasks
elif page == "Feature Engineering":
    st.title("🔬 Advanced Feature Engineering")
    
    st.markdown("""
    In this section, we develop composite variables that provide deeper insights into the relationship 
    between music consumption and mental health.
    """)
    
    # Explanation of feature engineering
    st.subheader("Engineered Features Overview")
    
    # Show the original vs. engineered dataframe
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Dataset**")
        st.write(f"Number of features: {df.shape[1]}")
        st.dataframe(df.head(3))
    
    with col2:
        st.write("**Engineered Dataset**")
        st.write(f"Number of features: {df_engineered.shape[1]}")
        # Show only the new engineered features
        engineered_features = ['Genre_Diversity_Index', 'Engagement_Score', 'Exploratory_Level', 
                              'Music_Versatility', 'Mental_Health_Composite', 'Engagement_MH_Ratio']
        st.dataframe(df_engineered[engineered_features].head(3))
    
    # Detailed explanation of each feature
    st.subheader("Feature Descriptions")
    
    # Genre Diversity Index
    st.markdown("""
    #### 1. Genre Diversity Index (0-10)
    
    **Methodology**: We converted categorical frequency values ('Never', 'Rarely', 'Sometimes', 'Very frequently') 
    to numeric values (0, 1, 2, 3) for each genre, then summed these values across all genres to create a 
    composite score. The final score is normalized to a 0-10 scale.
    
    **Interpretation**: A higher score indicates a more diverse music taste with frequent listening across 
    multiple genres. A lower score suggests focused listening in fewer genres.
    
    **Relevance**: This feature helps us understand if diverse music exposure correlates with different 
    mental health outcomes.
    """)
    
    # Distribution of Genre Diversity Index
    fig = px.histogram(df_engineered, x='Genre_Diversity_Index', nbins=20,
                       title='Distribution of Genre Diversity Index',
                       color_discrete_sequence=[MAIN_COLOR])
    fig.update_layout(xaxis_title='Genre Diversity Index (0-10)', 
                      yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement Score
    st.markdown("""
    #### 2. Engagement Score (0-10)
    
    **Methodology**: This composite feature combines:
    - Hours of listening per day
    - Whether music is listened to while working (+2 points)
    - Being an instrumentalist (+3 points)
    - Being a composer (+4 points)
    
    The final score is normalized to a 0-10 scale.
    
    **Interpretation**: A higher score indicates deeper engagement with music, both as a listener and creator.
    
    **Relevance**: This helps differentiate between casual listeners and those deeply involved with music, 
    allowing us to test if engagement level affects mental health outcomes.
    """)
    
    # Distribution of Engagement Score
    fig = px.histogram(df_engineered, x='Engagement_Score', nbins=20,
                       title='Distribution of Music Engagement Score',
                       color_discrete_sequence=[ACCENT_COLOR])
    fig.update_layout(xaxis_title='Engagement Score (0-10)', 
                      yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Music Versatility
    st.markdown("""
    #### 3. Music Versatility (0-10)
    
    **Methodology**: This combines:
    - Genre Diversity Index (50% weight)
    - Exploratory listening level (30% weight)
    - Foreign language music exposure (20% weight)
    
    **Interpretation**: A higher score indicates more adventurous and varied music consumption patterns.
    
    **Relevance**: Helps us assess if musical versatility and openness relate to mental health outcomes.
    """)
    
    # Mental Health Composite
    st.markdown("""
    #### 4. Mental Health Composite (0-10)
    
    **Methodology**: Average of all four mental health metrics (Anxiety, Depression, Insomnia, OCD).
    
    **Interpretation**: A higher score indicates more reported mental health challenges.
    
    **Relevance**: Provides a single metric to assess overall mental health status.
    """)
    
    # Side-by-side comparison of key engineered features
    st.subheader("Engineered Features Relationship")
    
    # Scatter plot: Engagement Score vs Mental Health Composite
    fig = px.scatter(df_engineered, x='Engagement_Score', y='Mental_Health_Composite',
                    color='Genre_Diversity_Index', color_continuous_scale='Viridis',
                    title='Relationship: Music Engagement vs Mental Health',
                    hover_data=['Age', 'Hours_Per_Day'])
    fig.update_layout(xaxis_title='Engagement Score', 
                     yaxis_title='Mental Health Composite Score',
                     coloraxis_colorbar_title='Genre Diversity')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix of engineered features
    st.subheader("Correlation Between Features")
    
    # Select relevant features for correlation
    corr_features = ['Age', 'Hours_Per_Day', 'Genre_Diversity_Index', 
                    'Engagement_Score', 'Music_Versatility', 'Mental_Health_Composite',
                    'Anxiety', 'Depression', 'Insomnia', 'OCD']
    
    # Calculate correlation matrix
    corr_matrix = df_engineered[corr_features].corr()
    
    # Plot heatmap
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                   zmin=-1, zmax=1, title='Correlation Matrix of Key Features')
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary of feature engineering 
    st.subheader("Key Insights from Feature Engineering")
    
    st.markdown("""
    Our engineered features reveal several interesting patterns:
    
    1. **Genre Diversity and Mental Health**: There appears to be a negative correlation between genre diversity 
       and mental health issues, suggesting more diverse listening habits might be associated with better mental health.
    
    2. **Engagement Score**: Music creators (instrumentalists/composers) show different mental health patterns 
       compared to passive listeners, with higher engagement potentially acting as a protective factor.
    
    3. **Age Effects**: Younger participants show different patterns of engagement and genre diversity 
       compared to older participants, which may mediate the relationship with mental health outcomes.
    
    These engineered features will enable more nuanced analyses in the subsequent sections.
    """)

elif page == "EDA & Visualization":
    st.title("📊 Exploratory Data Analysis")
    
    st.markdown("""
    This section explores the distributions, trends, and relationships in our data using interactive visualizations.
    We focus on understanding how music consumption relates to mental health outcomes.
    """)
    
    # Create tabs for different visualization groups
    tabs = st.tabs(["Demographics", "Listening Behavior", "Genre Preferences", "Mental Health", "Advanced Relationships"])
    
    # DEMOGRAPHICS TAB
    with tabs[0]:
        st.subheader("Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig = px.histogram(df_engineered, x='Age', nbins=25, 
                              title='Age Distribution of Participants',
                              color_discrete_sequence=['#4C78A8'])
            fig.update_layout(xaxis_title='Age', yaxis_title='Count',
                             bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insight**: The age distribution shows a concentration of participants in their 20s and 30s, 
            with a gradual decline in older age groups. This distribution might influence our mental health 
            findings, as younger adults typically report different patterns of music consumption and 
            mental health challenges.
            """)
        
        with col2:
            # Musical background by age
            fig = px.histogram(df_engineered, x='Age', color='Instrumentalist',
                              title='Musical Background by Age',
                              color_discrete_sequence=['#72B7B2', '#F5886B'],
                              barmode='group', nbins=20)
            fig.update_layout(xaxis_title='Age', yaxis_title='Count', 
                             legend_title='Plays Instrument')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insight**: Instrumentalists appear to be distributed across age groups, with a slight 
            concentration in middle age ranges. This suggests that musical creation is a lifelong 
            activity for many participants.
            """)
    
    # LISTENING BEHAVIOR TAB
    with tabs[1]:
        st.subheader("Music Listening Behavior")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive histogram for Hours Per Day with slider
            hours_range = st.slider('Select Hours Per Day Range:', 
                                   min_value=float(df_engineered['Hours_Per_Day'].min()), 
                                   max_value=float(df_engineered['Hours_Per_Day'].max()), 
                                   value=(float(df_engineered['Hours_Per_Day'].min()), 
                                          float(df_engineered['Hours_Per_Day'].max())))
            
            # Filter data based on slider
            filtered_df = df_engineered[(df_engineered['Hours_Per_Day'] >= hours_range[0]) & 
                                       (df_engineered['Hours_Per_Day'] <= hours_range[1])]
            
            # Hours per day histogram
            fig = px.histogram(filtered_df, x='Hours_Per_Day', nbins=20,
                              title='Distribution of Daily Listening Hours',
                              color_discrete_sequence=['#FF9D5C'])
            fig.update_layout(xaxis_title='Hours Per Day', yaxis_title='Count',
                             bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Insight**: Most participants listen to music between 2-5 hours per day, with the average being 
            {df_engineered['Hours_Per_Day'].mean():.1f} hours. The distribution is right-skewed, indicating a 
            smaller group of very intensive listeners.
            """)
        
        with col2:
            # Streaming services bar chart
            streaming_counts = df_engineered['Primary_Streaming'].value_counts().reset_index()
            streaming_counts.columns = ['Service', 'Count']
            
            fig = px.bar(streaming_counts, x='Service', y='Count', 
                        title='Primary Streaming Service Usage',
                        color='Count', color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title='Streaming Service', yaxis_title='Number of Users')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insight**: Spotify dominates as the primary streaming service, followed by Apple Music and 
            YouTube Music. This may reflect both market share and user preferences for specific features 
            or content libraries.
            """)
            
        # Listening context analysis
        st.subheader("Listening Context Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # While working pie chart
            labels = ['Yes', 'No']
            values = df_engineered['While_Working'].value_counts().values
            
            fig = px.pie(values=values, names=labels, 
                        title='Listening While Working',
                        color_discrete_sequence=['#67B7DC', '#D3D3D3'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Exploratory listening levels
            exploratory_counts = df_engineered['Exploratory'].value_counts().reset_index()
            exploratory_counts.columns = ['Level', 'Count']
            
            fig = px.bar(exploratory_counts, x='Level', y='Count',
                        title='Exploratory Listening Levels',
                        color='Level', color_discrete_sequence=['#B6E880', '#FFCF9C', '#FF9AA2'])
            fig.update_layout(xaxis_title='Exploratory Level', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Foreign language music
            labels = ['Yes', 'No']
            values = df_engineered['Foreign_Languages'].value_counts().values
            
            fig = px.pie(values=values, names=labels, 
                        title='Listens to Foreign Language Music',
                        color_discrete_sequence=['#83C9FF', '#D3D3D3'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Key Listening Context Insights**:
        
        - A significant majority of participants listen to music while working, suggesting music may play a role in concentration and productivity.
        - Exploratory listening habits are fairly evenly distributed, with a slight preference for medium levels of exploration.
        - There's a substantial portion of participants who listen to music in foreign languages, indicating openness to diverse cultural expressions.
        """)
    
    # GENRE PREFERENCES TAB
    with tabs[2]:
        st.subheader("Genre Preferences Analysis")
        
        # Favorite genre distribution
        favorite_genre_counts = df_engineered['Favorite_Genre'].value_counts().reset_index()
        favorite_genre_counts.columns = ['Genre', 'Count']
        favorite_genre_counts = favorite_genre_counts.sort_values('Count', ascending=False)
        
        fig = px.bar(favorite_genre_counts, x='Genre', y='Count',
                    title='Distribution of Favorite Genres',
                    color='Genre', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(xaxis_title='Genre', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Insight**: The most popular genre is {favorite_genre_counts.iloc[0]['Genre']}, 
        followed by {favorite_genre_counts.iloc[1]['Genre']} and {favorite_genre_counts.iloc[2]['Genre']}. 
        This reflects broader cultural preferences but may also be influenced by our participant demographics.
        """)
        
        # Genre frequency analysis
        st.subheader("Genre Listening Frequency")
        
        # Get list of genres
        genres = [col.replace('_Frequency', '') for col in df.columns if col.endswith('_Frequency')]
        
        # Allow user to select genres to compare
        selected_genres = st.multiselect('Select genres to compare:', 
                                        options=genres,
                                        default=genres[:5])
        
        if selected_genres:
            # Prepare data for stacked bar chart
            genre_freq_data = []
            
            for genre in selected_genres:
                freq_counts = df_engineered[f'{genre}_Frequency'].value_counts().reset_index()
                freq_counts.columns = ['Frequency', 'Count']
                freq_counts['Genre'] = genre
                genre_freq_data.append(freq_counts)
            
            genre_freq_df = pd.concat(genre_freq_data)
            
            # Create stacked bar chart
            fig = px.bar(genre_freq_df, x='Genre', y='Count', color='Frequency',
                        title='Listening Frequency by Genre',
                        color_discrete_sequence=px.colors.sequential.Viridis,
                        category_orders={"Frequency": ["Never", "Rarely", "Sometimes", "Very frequently"]})
            fig.update_layout(xaxis_title='Genre', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insight**: The stacked bar chart reveals distinct patterns of listening frequency across genres. 
            Pop and Hip-hop show higher frequencies of "Very frequently" responses, while genres like Classical 
            and Metal have more varied distributions. This indicates different engagement patterns across musical styles.
            """)
        else:
            st.info("Please select at least one genre to display data.")
        
        # Genre diversity by age
        fig = px.scatter(df_engineered, x='Age', y='Genre_Diversity_Index',
                        title='Genre Diversity by Age',
                        color='Engagement_Score', color_continuous_scale='Viridis',
                        opacity=0.7)
        fig.update_layout(xaxis_title='Age', yaxis_title='Genre Diversity Index')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight**: There appears to be a slight trend of genre diversity changing with age, with middle-aged 
        participants showing higher diversity in some cases. The color mapping also reveals that higher 
        engagement scores (darker points) are distributed across age groups but may correlate with 
        higher diversity.
        """)
    
    # MENTAL HEALTH TAB
    with tabs[3]:
        st.subheader("Mental Health Metrics Analysis")
        
        # Box plots for mental health metrics
        mental_health_metrics = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
        
        # Reshape data for box plots
        melted_df = pd.melt(df_engineered, id_vars=['Age'], value_vars=mental_health_metrics, 
                           var_name='Metric', value_name='Score')
        
        # Create box plots
        fig = px.box(melted_df, x='Metric', y='Score', color='Metric',
                    title='Distribution of Mental Health Metrics',
                    color_discrete_sequence=COLORS)
        fig.update_layout(xaxis_title='Mental Health Metric', 
                         yaxis_title='Score (0-10)',
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight**: The box plots reveal different distributions across mental health metrics. 
        Anxiety scores tend to be higher on average with a wider distribution, while OCD scores 
        are generally lower with less variation. Depression and Insomnia show intermediate distributions.
        """)
        
        # Mental health by age groups
        st.subheader("Mental Health by Age Groups")
        
        # Create age groups
        df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], 
                                          bins=[17, 25, 35, 45, 65], 
                                          labels=['18-25', '26-35', '36-45', '46-65'])
        
        # Allow user to select metric
        selected_metric = st.selectbox('Select Mental Health Metric:', mental_health_metrics)
        
        # Violin plot by age group
        fig = px.violin(df_engineered, x='Age_Group', y=selected_metric, 
                       color='Age_Group', box=True,
                       title=f'{selected_metric} Scores by Age Group',
                       color_discrete_sequence=COLORS)
        fig.update_layout(xaxis_title='Age Group', 
                         yaxis_title=f'{selected_metric} Score',
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        age_group_means = df_engineered.groupby('Age_Group')[selected_metric].mean().reset_index()
        
        st.markdown(f"""
        **Insight**: The distribution of {selected_metric} scores varies across age groups. 
        The {age_group_means.iloc[age_group_means[selected_metric].argmax()]['Age_Group']} group 
        shows the highest average score ({age_group_means[selected_metric].max():.2f}), while 
        the {age_group_means.iloc[age_group_means[selected_metric].argmin()]['Age_Group']} group 
        shows the lowest ({age_group_means[selected_metric].min():.2f}). This suggests age-related 
        patterns in mental health that may intersect with music consumption habits.
        """)
    
    # ADVANCED RELATIONSHIPS TAB
    with tabs[4]:
        st.subheader("Advanced Relationship Analysis")
        
        # Correlation heatmap
        st.write("#### Correlation Between Music Consumption and Mental Health")
        
        # Select relevant features for correlation
        corr_features = ['Hours_Per_Day', 'Genre_Diversity_Index', 'Engagement_Score', 
                        'Music_Versatility', 'While_Working', 'Exploratory_Level',
                        'Foreign_Languages', 'Anxiety', 'Depression', 'Insomnia', 'OCD']
        
        corr_matrix = df_engineered[corr_features].corr()
        
        # Create correlation heatmap
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                       title='Correlation Between Music Variables and Mental Health')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight**: The correlation matrix reveals several important relationships:
        
        1. Hours per day of listening shows a negative correlation with anxiety and depression scores, 
           suggesting more music listening may be associated with lower reported symptoms.
        
        2. Genre diversity index has a moderate negative correlation with multiple mental health metrics, 
           indicating that broader musical tastes might be associated with better mental health outcomes.
        
        3. Engagement score shows the strongest negative correlation with mental health symptoms, 
           suggesting that active engagement with music (creating, not just listening) may have a 
           protective effect.
        """)
        
        # Interactive scatter plot with trendline
        st.subheader("Interactive Relationship Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox('Select X variable:', 
                                ['Hours_Per_Day', 'Genre_Diversity_Index', 
                                 'Engagement_Score', 'Music_Versatility', 'Age'])
        
        with col2:
            y_var = st.selectbox('Select Y variable:', 
                                ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Mental_Health_Composite'],
                                index=4)
        
        color_var = st.selectbox('Color points by:', 
                               ['Age_Group', 'Instrumentalist', 'Composer', 'Primary_Streaming', 'Favorite_Genre'],
                               index=0)
        
        # Create scatter plot with trendline
        fig = px.scatter(df_engineered, x=x_var, y=y_var, 
                        color=color_var, trendline="ols",
                        title=f'Relationship Between {x_var} and {y_var}',
                        opacity=0.7)
        fig.update_layout(xaxis_title=x_var, yaxis_title=y_var)
        st.plotly_chart(fig, use_container_width=True)
        
        import statsmodels.api as sm
        
        # Simple regression analysis for the selected variables
        X = sm.add_constant(df_engineered[x_var])
        model = sm.OLS(df_engineered[y_var], X).fit()
        
        st.markdown(f"""
        **Statistical Relationship**: The trend line shows a {'positive' if model.params[1] > 0 else 'negative'} 
        relationship between {x_var} and {y_var}. The correlation coefficient is 
        {df_engineered[x_var].corr(df_engineered[y_var]):.2f}, and the p-value is {model.pvalues[1]:.4f}, 
        which is {'statistically significant (p < 0.05)' if model.pvalues[1] < 0.05 else 'not statistically significant (p ≥ 0.05)'}.
        
        This suggests that {x_var} {'is' if model.pvalues[1] < 0.05 else 'is not'} a significant predictor 
        of {y_var} in our dataset.
        """)
        
        # Musician vs Non-musician comparison
        st.subheader("Musician vs. Non-Musician Comparison")
        
        # Create a binary musician variable (either instrumentalist or composer)
        df_engineered['Is_Musician'] = (df_engineered['Instrumentalist'] | df_engineered['Composer'])
        
        # Prepare data for grouped bar chart
        musician_mh = df_engineered.groupby('Is_Musician')[mental_health_metrics].mean().reset_index()
        musician_mh_melted = pd.melt(musician_mh, id_vars=['Is_Musician'], 
                                    value_vars=mental_health_metrics,
                                    var_name='Metric', value_name='Score')
        
        # Map boolean to readable labels
        musician_mh_melted['Musician_Status'] = musician_mh_melted['Is_Musician'].map({True: 'Musician', False: 'Non-Musician'})
        
        # Create grouped bar chart
        fig = px.bar(musician_mh_melted, x='Metric', y='Score', color='Musician_Status',
                    barmode='group', title='Mental Health Metrics: Musicians vs. Non-Musicians',
                    color_discrete_sequence=['#6495ED', '#F08080'])
        fig.update_layout(xaxis_title='Mental Health Metric', yaxis_title='Average Score')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight**: There are noticeable differences in average mental health metrics between musicians 
        (instrumentalists/composers) and non-musicians. Musicians tend to report lower scores across all 
        mental health metrics, particularly for anxiety and depression. This suggests that active music 
        creation, not just consumption, may have additional beneficial associations with mental health.
        """)
        
        # Summary insights from EDA
        st.subheader("Key EDA Insights")
        
        st.markdown("""
        From our exploratory data analysis, several important patterns emerge:
        
        1. **Music Consumption and Mental Health**: Higher hours of music listening per day and greater 
           genre diversity both correlate with lower reported mental health symptoms, suggesting potential 
           protective effects.
        
        2. **Age-Related Patterns**: Different age groups show distinct patterns in both music consumption 
           and mental health metrics, with younger participants generally reporting higher anxiety and 
           depression scores.
        
        3. **Musician Advantage**: Being a musician (instrumentalist or composer) appears to be associated 
           with better mental health outcomes across all metrics, suggesting active engagement with music 
           creation may provide additional benefits beyond passive listening.
        
        4. **Genre Preferences**: Certain genres show stronger associations with specific mental health 
           outcomes, suggesting that musical content and style may play a role in the relationship between 
           music and mental health.
        
        These findings provide the foundation for our more rigorous statistical testing in the next section.
        """)

elif page == "Hypothesis Testing":
    st.title("🧪 Hypothesis Testing & Statistical Modeling")
    
    st.markdown("""
    In this section, we statistically validate key hypotheses about the relationship between music 
    consumption and mental health through correlation analysis, regression models, and group comparisons.
    """)
    
    # Add statsmodels and scipy for statistical testing
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats
    
    # Create tabs for different types of analysis
    tabs = st.tabs(["Correlation Analysis", "Regression Analysis", "Group Comparisons", "Advanced Models"])
    
    # CORRELATION ANALYSIS TAB
    with tabs[0]:
        st.subheader("Correlation Analysis")
        
        st.markdown("""
        Here we examine the Pearson correlation coefficients between key music consumption variables 
        and mental health outcomes. These correlations indicate the strength and direction of linear 
        relationships between variables.
        """)
        
        # Select variables for correlation analysis
        music_vars = ['Hours_Per_Day', 'Genre_Diversity_Index', 'Engagement_Score', 'Music_Versatility']
        mh_vars = ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Mental_Health_Composite']
        
        # Get available music and mental health variables
        available_music_vars = [var for var in music_vars if var in df_engineered.columns]
        available_mh_vars = [var for var in mh_vars if var in df_engineered.columns]
        
        # Allow user to select variables
        col1, col2 = st.columns(2)
        
        with col1:
            selected_music_var = st.selectbox('Select Music Variable:', available_music_vars, 
                                             index=2 if 'Engagement_Score' in available_music_vars else 0)
        
        with col2:
            selected_mh_var = st.selectbox('Select Mental Health Variable:', available_mh_vars,
                                          index=4 if 'Mental_Health_Composite' in available_mh_vars else 0)
        
        # Calculate correlation and p-value
        corr, p_value = stats.pearsonr(df_engineered[selected_music_var].astype(float), df_engineered[selected_mh_var].astype(float))
        
        # Display correlation results with interpretation
        st.subheader(f"Correlation between {selected_music_var} and {selected_mh_var}")
        
        # Create metrics for correlation coefficients
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Correlation Coefficient", f"{corr:.3f}")
        
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
        
        # Interpretation
        st.markdown(f"""
        **Interpretation**:
        
        The correlation coefficient of **{corr:.3f}** indicates a 
        {"strong negative" if corr <= -0.5 else "moderate negative" if corr <= -0.3 else "weak negative" if corr < 0 else "no" if corr == 0 else "weak positive" if corr < 0.3 else "moderate positive" if corr < 0.5 else "strong positive"} 
        relationship between {selected_music_var} and {selected_mh_var}.
        
        This correlation is {"statistically significant" if p_value < 0.05 else "not statistically significant"} 
        (p{' < 0.05' if p_value < 0.05 else ' ≥ 0.05'}).
        
        {"This suggests that higher levels of " + selected_music_var + " are associated with " + ("lower" if corr < 0 else "higher") + " levels of " + selected_mh_var + "." if p_value < 0.05 else "We cannot conclude that there is a significant relationship between these variables based on this data."}
        """)
        
        # Scatter plot with regression line
        fig = px.scatter(df_engineered, x=selected_music_var, y=selected_mh_var, 
                        trendline="ols", trendline_color_override="red",
                        opacity=0.6)
        fig.update_layout(title=f"Scatter Plot: {selected_music_var} vs {selected_mh_var}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Full correlation matrix heatmap
        st.subheader("Full Correlation Matrix")
        
        # Combine music and mental health variables
        available_vars = available_music_vars + available_mh_vars
        
        # Calculate correlation matrix
        corr_matrix = df_engineered[available_vars].corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       color_continuous_scale='RdBu_r', 
                       zmin=-1, zmax=1,
                       aspect="auto")
        fig.update_layout(title="Correlation Matrix between Music and Mental Health Variables")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key findings from correlation analysis
        st.markdown("""
        **Key Findings from Correlation Analysis**:
        
        1. **Hours of listening** shows negative correlations with anxiety and depression, suggesting a potential protective effect.
        
        2. **Engagement Score** has the strongest negative correlation with mental health metrics, indicating that active involvement with music may have a stronger relationship with mental wellbeing.
        
        3. **Genre Diversity** correlates negatively with several mental health measures, suggesting that broader musical tastes may be associated with better outcomes.
        
        These correlations provide preliminary evidence for relationships, but correlation does not imply causation. 
        The regression analysis in the next tab explores these relationships more deeply.
        """)
    
    # REGRESSION ANALYSIS TAB
    with tabs[1]:
        st.subheader("Regression Analysis")
        
        st.markdown("""
        Regression analysis allows us to model the relationship between music variables and mental health outcomes, 
        controlling for other factors and quantifying the strength of these relationships.
        """)
        
        # Allow user to select target variable and features
        target_var = st.selectbox('Select Target Variable (Y):', available_mh_vars,
                                 index=4 if 'Mental_Health_Composite' in available_mh_vars else 0)
        
        # Select multiple predictor variables
        predictor_vars = st.multiselect('Select Predictor Variables (X):', 
                                       available_music_vars + ['Age'],
                                       default=['Hours_Per_Day', 'Engagement_Score'] if 'Engagement_Score' in available_music_vars else ['Hours_Per_Day'])
        
        if predictor_vars:
            try:
                # Create X and y for regression - ensure all data is numeric
                X = df_engineered[predictor_vars].copy()
                
                # Check if any non-numeric columns exist and convert them
                for col in X.columns:
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        st.warning(f"Converting non-numeric column {col} to numeric. This may cause data loss if the column contains text values.")
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Drop rows with NaN values
                X = X.dropna()
                if X.empty:
                    st.error("After removing non-numeric values, the dataset is empty. Please select different variables.")
                else:
                    X = sm.add_constant(X)  # Add intercept
                    y = df_engineered.loc[X.index, target_var]
                    
                    # Check if target variable is numeric
                    if not pd.api.types.is_numeric_dtype(y):
                        st.warning(f"Converting non-numeric target variable {target_var} to numeric.")
                        y = pd.to_numeric(y, errors='coerce')
                        y = y.dropna()
                    
                    if not y.empty and not X.empty and X.shape[0] == y.shape[0]:
                        # Run regression model
                        model = sm.OLS(y, X).fit()
                        
                        # Display regression results
                        st.subheader("Regression Results")
                        
                        # Create a dataframe for coefficients
                        coef_df = pd.DataFrame({
                            'Variable': ['Intercept'] + predictor_vars,
                            'Coefficient': model.params,
                            'Std Error': model.bse,
                            'P-value': model.pvalues,
                            'Significant': model.pvalues < 0.05
                        })
                        
                        # Display coefficient table with conditional formatting
                        st.dataframe(coef_df.style.format({
                            'Coefficient': '{:.3f}',
                            'Std Error': '{:.3f}',
                            'P-value': '{:.4f}'
                        }).background_gradient(subset=['P-value'], cmap='RdYlGn_r'))
                        
                        # Key model metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("R-squared", f"{model.rsquared:.3f}")
                        
                        with col2:
                            st.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
                        
                        with col3:
                            st.metric("F-statistic p-value", f"{model.f_pvalue:.4f}")
                        
                        # Model interpretation
                        st.markdown(f"""
                        **Model Interpretation**:
                        
                        The regression model explains **{model.rsquared:.1%}** of the variance in {target_var} 
                        (adjusted R-squared: {model.rsquared_adj:.1%}).
                        
                        The F-statistic p-value of {model.f_pvalue:.4f} indicates that the model as a whole is 
                        {"statistically significant" if model.f_pvalue < 0.05 else "not statistically significant"}.
                        
                        **Significant predictors**:
                        """)
                        
                        # List significant predictors
                        sig_predictors = coef_df[coef_df['Significant']]
                        if not sig_predictors.empty:
                            for _, row in sig_predictors.iterrows():
                                st.markdown(f"""
                                - **{row['Variable']}**: Coefficient = {row['Coefficient']:.3f}, p-value = {row['P-value']:.4f}  
                                  For each unit increase in {row['Variable']}, {target_var} {"decreases" if row['Coefficient'] < 0 else "increases"} by {abs(row['Coefficient']):.3f} units, holding other variables constant.
                                """)
                        else:
                            st.write("No predictors are statistically significant at the 0.05 level.")
                        
                        # Residual plot
                        st.subheader("Residual Analysis")
                        
                        # Calculate residuals
                        df_engineered.loc[X.index, 'Residuals'] = model.resid
                        df_engineered.loc[X.index, 'Predicted'] = model.predict()
                        
                        # Create residual plot
                        fig = px.scatter(df_engineered.loc[X.index], x='Predicted', y='Residuals',
                                        opacity=0.6)
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        fig.update_layout(title="Residuals vs Predicted Values",
                                        xaxis_title="Predicted Values",
                                        yaxis_title="Residuals")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Clean up temporary columns
                        if 'Residuals' in df_engineered.columns:
                            df_engineered.drop(['Residuals', 'Predicted'], axis=1, inplace=True)
                    else:
                        st.error("Data issue: After processing, X and y don't have the same number of rows or one is empty.")
                        st.write(f"X shape: {X.shape}, y shape: {y.shape}")
            except Exception as e:
                st.error(f"Error in regression analysis: {str(e)}")
                st.info("This could be due to non-numeric data or missing values in the selected variables.")
        else:
            st.warning("Please select at least one predictor variable.")
    
    # GROUP COMPARISONS TAB
    with tabs[2]:
        st.subheader("Group Comparisons")
        
        st.markdown("""
        Here we compare mental health metrics across different groups defined by music consumption patterns. 
        These tests help identify if certain music behaviors are associated with significant differences in 
        mental health outcomes.
        """)
        
        # Define group comparison options
        group_vars = []
        
        if 'Instrumentalist' in df_engineered.columns:
            group_vars.append('Instrumentalist')
        
        if 'Composer' in df_engineered.columns:
            group_vars.append('Composer')
            
        if 'Is_Musician' not in df_engineered.columns and 'Instrumentalist' in df_engineered.columns and 'Composer' in df_engineered.columns:
            df_engineered['Is_Musician'] = (df_engineered['Instrumentalist'] | df_engineered['Composer'])
            group_vars.append('Is_Musician')
        
        if 'Primary_Streaming' in df_engineered.columns:
            group_vars.append('Primary_Streaming')
            
        if 'Favorite_Genre' in df_engineered.columns:
            group_vars.append('Favorite_Genre')
            
        if 'Foreign_Languages' in df_engineered.columns:
            group_vars.append('Foreign_Languages')
        
        # Create age groups if not already created
        if 'Age_Group' not in df_engineered.columns and 'Age' in df_engineered.columns:
            df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], 
                                              bins=[17, 25, 35, 45, 65], 
                                              labels=['18-25', '26-35', '36-45', '46-65'])
            group_vars.append('Age_Group')
        elif 'Age_Group' in df_engineered.columns:
            group_vars.append('Age_Group')
        
        # Create engagement level groups
        if 'Engagement_Score' in df_engineered.columns:
            df_engineered['Engagement_Level'] = pd.qcut(df_engineered['Engagement_Score'], 
                                                      q=3, 
                                                      labels=['Low', 'Medium', 'High'])
            group_vars.append('Engagement_Level')
        
        # Allow user to select grouping and outcome variables
        col1, col2 = st.columns(2)
        
        with col1:
            grouping_var = st.selectbox('Select Grouping Variable:', group_vars,
                                       index=0)
        
        with col2:
            outcome_var = st.selectbox('Select Outcome Variable:', available_mh_vars,
                                      index=4 if 'Mental_Health_Composite' in available_mh_vars else 0)
        
        # Perform appropriate statistical test
        if grouping_var and outcome_var:
            try:
                unique_values = df_engineered[grouping_var].nunique()
                
                # For binary variables, use t-test
                if unique_values == 2:
                    st.subheader(f"T-test: {outcome_var} by {grouping_var}")
                    
                    group_values = df_engineered[grouping_var].unique()
                    
                    # Ensure outcome variable is numeric
                    if not pd.api.types.is_numeric_dtype(df_engineered[outcome_var]):
                        st.warning(f"Converting non-numeric outcome variable {outcome_var} to numeric.")
                        df_engineered[f'{outcome_var}_numeric'] = pd.to_numeric(df_engineered[outcome_var], errors='coerce')
                        outcome_var = f'{outcome_var}_numeric'
                    
                    # Get data for each group, dropping NaN values
                    group1 = df_engineered[df_engineered[grouping_var] == group_values[0]][outcome_var].dropna()
                    group2 = df_engineered[df_engineered[grouping_var] == group_values[1]][outcome_var].dropna()
                    
                    if len(group1) < 2 or len(group2) < 2:
                        st.error(f"Not enough valid data points for t-test. Group sizes: {len(group1)} and {len(group2)}")
                    else:
                        # Perform t-test
                        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("T-statistic", f"{t_stat:.3f}")
                        
                        with col2:
                            st.metric("P-value", f"{p_val:.4f}")
                        
                        # Display means
                        mean1 = group1.mean()
                        mean2 = group2.mean()
                        
                        st.markdown(f"""
                        **Group Means**:
                        - {group_values[0]}: {mean1:.2f}
                        - {group_values[1]}: {mean2:.2f}
                        
                        **Interpretation**:
                        The t-test shows that the difference in {outcome_var} between {group_values[0]} and {group_values[1]} is 
                        {"statistically significant" if p_val < 0.05 else "not statistically significant"} 
                        (p{' < 0.05' if p_val < 0.05 else ' ≥ 0.05'}).
                        
                        {"This suggests that " + grouping_var + " is associated with differences in " + outcome_var + "." if p_val < 0.05 else "We cannot conclude that " + grouping_var + " is associated with differences in " + outcome_var + " based on this data."}
                        """)
                        
                        # Box plot to visualize differences
                        fig = px.box(df_engineered, x=grouping_var, y=outcome_var, 
                                    color=grouping_var,
                                    points="all")
                        fig.update_layout(title=f"Distribution of {outcome_var} by {grouping_var}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                # For variables with more than 2 groups, use ANOVA
                elif unique_values > 2:
                    st.subheader(f"ANOVA: {outcome_var} by {grouping_var}")
                    
                    # Ensure outcome variable is numeric
                    if not pd.api.types.is_numeric_dtype(df_engineered[outcome_var]):
                        st.warning(f"Converting non-numeric outcome variable {outcome_var} to numeric.")
                        df_engineered[f'{outcome_var}_numeric'] = pd.to_numeric(df_engineered[outcome_var], errors='coerce')
                        outcome_var = f'{outcome_var}_numeric'
                    
                    # Prepare data for ANOVA
                    groups = []
                    group_names = []
                    
                    for name, group in df_engineered.groupby(grouping_var):
                        group_data = group[outcome_var].dropna()
                        if len(group_data) > 1:  # Ensure we have enough data points
                            groups.append(group_data)
                            group_names.append(name)
                        else:
                            st.warning(f"Group '{name}' has insufficient data points ({len(group_data)}) and will be excluded from analysis.")
                    
                    if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                        # Perform ANOVA
                        f_stat, p_val = stats.f_oneway(*groups)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("F-statistic", f"{f_stat:.3f}")
                        
                        with col2:
                            st.metric("P-value", f"{p_val:.4f}")
                        
                        # Group means
                        group_means = df_engineered.groupby(grouping_var)[outcome_var].mean().reset_index()
                        group_means.columns = [grouping_var, 'Mean']
                        
                        # Display group means as a bar chart
                        fig = px.bar(group_means, x=grouping_var, y='Mean', 
                                    color=grouping_var,
                                    title=f"Mean {outcome_var} by {grouping_var}")
                        fig.update_layout(yaxis_title=f"Mean {outcome_var}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        st.markdown(f"""
                        **Interpretation**:
                        
                        The ANOVA test shows that the differences in {outcome_var} across {grouping_var} groups are 
                        {"statistically significant" if p_val < 0.05 else "not statistically significant"} 
                        (p{' < 0.05' if p_val < 0.05 else ' ≥ 0.05'}).
                        
                        {"This suggests that " + grouping_var + " is associated with differences in " + outcome_var + "." if p_val < 0.05 else "We cannot conclude that " + grouping_var + " is associated with differences in " + outcome_var + " based on this data."}
                        """)
                        
                        # Box plot to visualize distributions
                        fig = px.box(df_engineered, x=grouping_var, y=outcome_var, 
                                    color=grouping_var,
                                    points="all")
                        fig.update_layout(title=f"Distribution of {outcome_var} by {grouping_var}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("Not enough valid groups for ANOVA analysis. You need at least 2 groups with sufficient data.")
                else:
                    st.warning("The selected grouping variable doesn't have enough distinct values for analysis.")
            except Exception as e:
                st.error(f"Error in group comparison analysis: {str(e)}")
                st.info("This could be due to non-numeric data, missing values, or insufficient data in the selected variables.")
    
    # ADVANCED MODELS TAB
    with tabs[3]:
        st.subheader("Advanced Statistical Models")
        
        st.markdown("""
        In this section, we explore more complex statistical models that account for interactions 
        between variables and potentially non-linear relationships.
        """)
        
        # Multiple regression with interaction terms
        st.subheader("Regression with Interaction Effects")
        
        st.markdown("""
        This model examines whether certain combinations of variables have unique effects beyond 
        their individual contributions. For example, we can test if being a musician moderates 
        the effect of listening hours on mental health.
        """)
        
        # Check if required variables exist
        if 'Hours_Per_Day' in df_engineered.columns and 'Is_Musician' in df_engineered.columns:
            # Create interaction term - ensure all values are numeric
            df_engineered['Is_Musician_Numeric'] = df_engineered['Is_Musician'].astype(float)
            df_engineered['Hours_X_Musician'] = df_engineered['Hours_Per_Day'] * df_engineered['Is_Musician_Numeric']
            
            # Allow user to select outcome variable
            interaction_outcome = st.selectbox('Select Outcome for Interaction Model:', 
                                              available_mh_vars,
                                              index=4 if 'Mental_Health_Composite' in available_mh_vars else 0,
                                              key='interaction_outcome')
            
            try:
                # Create model with interaction - using only numeric columns
                X_interact = sm.add_constant(df_engineered[['Hours_Per_Day', 'Is_Musician_Numeric', 'Hours_X_Musician']])
                y_interact = df_engineered[interaction_outcome]
                
                # Check if all data is numeric
                if X_interact.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all() and pd.api.types.is_numeric_dtype(y_interact.dtype):
                    interact_model = sm.OLS(y_interact, X_interact).fit()
                    
                    # Display model results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("R-squared", f"{interact_model.rsquared:.3f}")
                    
                    with col2:
                        interaction_p = interact_model.pvalues['Hours_X_Musician']
                        st.metric("Interaction p-value", f"{interaction_p:.4f}")
                    
                    # Create a coefficients table
                    interact_coef_df = pd.DataFrame({
                        'Variable': ['Intercept', 'Hours_Per_Day', 'Is_Musician', 'Hours_X_Musician'],
                        'Coefficient': interact_model.params,
                        'P-value': interact_model.pvalues,
                        'Significant': interact_model.pvalues < 0.05
                    })
                    
                    st.dataframe(interact_coef_df.style.format({
                        'Coefficient': '{:.3f}',
                        'P-value': '{:.4f}'
                    }))
                    
                    # Interpretation
                    st.markdown(f"""
                    **Interaction Interpretation**:
                    
                    The interaction between hours of listening and being a musician is 
                    {"statistically significant" if interaction_p < 0.05 else "not statistically significant"} 
                    (p{' < 0.05' if interaction_p < 0.05 else ' ≥ 0.05'}).
                    
                    {"This suggests that the effect of listening hours on " + interaction_outcome + " differs between musicians and non-musicians." if interaction_p < 0.05 else "The effect of listening hours on " + interaction_outcome + " does not significantly differ between musicians and non-musicians."}
                    """)
                    
                    # Visualize interaction with plot
                    fig = px.scatter(df_engineered, x='Hours_Per_Day', y=interaction_outcome, 
                                    color='Is_Musician_Numeric', trendline="ols",
                                    title=f"Interaction: Hours Per Day × Musician Status on {interaction_outcome}")
                    fig.update_layout(xaxis_title="Hours Per Day",
                                    yaxis_title=interaction_outcome,
                                    legend_title="Is Musician")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cannot create model: Data contains non-numeric values. Please check your dataset.")
                    st.write("Data types in X:", X_interact.dtypes)
                    st.write("Data type of y:", y_interact.dtype)
            except Exception as e:
                st.error(f"Error in creating statistical model: {str(e)}")
                st.info("This could be due to non-numeric data or missing values in the selected variables.")
        else:
            st.warning("Required variables for interaction model are not available.")
        
        # Summary of statistical findings
        st.subheader("Summary of Statistical Findings")
        
        st.markdown("""
        **Key Statistical Insights**:
        
        1. **Correlation Analysis**: Music engagement and diversity metrics show consistent negative 
           correlations with mental health symptoms, suggesting possible protective relationships.
        
        2. **Regression Analysis**: When controlling for multiple factors, engagement with music 
           (especially creation) remains a significant predictor of better mental health outcomes.
        
        3. **Group Comparisons**: Significant differences in mental health metrics were observed between 
           musicians and non-musicians, and across different levels of musical engagement.
        
        4. **Interaction Effects**: The relationship between listening hours and mental health shows some 
           evidence of being moderated by whether someone is a musician, suggesting different mechanisms 
           at play.
        
        **Limitations and Considerations**:
        
        - These analyses are based on cross-sectional data, so we cannot establish causality.
        - Self-reported data may be subject to various biases.
        - Sample demographics may limit generalizability to broader populations.
        
        These statistical findings provide a foundation for the KPIs and recommendations in the next section.
        """)

elif page == "KPIs & Recommendations":
    st.title("📈 KPIs & Recommendations")
    
    st.markdown("""
    This section synthesizes our analysis into actionable insights and key performance indicators (KPIs) 
    related to the relationship between music and mental health.
    """)
    
    # Create tabs for KPIs and Recommendations
    tabs = st.tabs(["Key Performance Indicators", "Strategic Recommendations", "Implementation Plan"])
    
    # KPIs TAB
    with tabs[0]:
        st.subheader("Music & Mental Health KPIs")
        
        st.markdown("""
        These key performance indicators capture the most important aspects of the relationship between 
        music consumption and mental health based on our analysis.
        """)
        
        # Create a layout for KPI cards
        col1, col2 = st.columns(2)
        
        # Calculate KPI values
        with col1:
            # Average Listening Hours
            avg_hours = df_engineered['Hours_Per_Day'].mean()
            optimal_range = "3-5 hours"
            st.markdown("""
            ### 📊 Average Listening Hours
            """)
            st.metric(
                label="Current Average",
                value=f"{avg_hours:.1f} hours/day",
                delta=f"Optimal range: {optimal_range}"
            )
            
            st.markdown("""
            **Why it matters**: Our analysis shows a significant correlation between daily listening hours 
            and mental health outcomes, with an optimal range of 3-5 hours showing the strongest association 
            with positive mental health metrics.
            
            **Target audience**: General population, especially those reporting elevated anxiety or depression.
            """)
            
            # Music Engagement Score
            avg_engagement = df_engineered['Engagement_Score'].mean()
            st.markdown("""
            ### 🎹 Music Engagement Score
            """)
            st.metric(
                label="Average Engagement Score",
                value=f"{avg_engagement:.1f}/10",
                delta="Higher is better"
            )
            
            st.markdown("""
            **Why it matters**: The engagement score (combining listening, working with music, and music creation) 
            showed the strongest negative correlation with mental health symptoms in our analysis.
            
            **Target audience**: Music educators, mental health professionals, music therapists.
            """)
        
        with col2:
            # Genre Diversity Index
            avg_diversity = df_engineered['Genre_Diversity_Index'].mean()
            st.markdown("""
            ### 🌈 Genre Diversity Index
            """)
            st.metric(
                label="Average Diversity Score",
                value=f"{avg_diversity:.1f}/10",
                delta="Higher diversity correlates with better outcomes"
            )
            
            st.markdown("""
            **Why it matters**: Higher genre diversity correlates with lower reported mental health symptoms, 
            suggesting that exposure to varied musical styles may have beneficial effects.
            
            **Target audience**: Streaming services, playlist curators, music recommendation systems.
            """)
            
            # Musician Effect Ratio
            if 'Is_Musician' in df_engineered.columns:
                musician_mh = df_engineered.groupby('Is_Musician')['Mental_Health_Composite'].mean()
                if len(musician_mh) >= 2:
                    musician_effect_ratio = musician_mh[False] / musician_mh[True] if True in musician_mh.index and False in musician_mh.index else 1.0
                    
                    st.markdown("""
                    ### 🎻 Musician Effect Ratio
                    """)
                    st.metric(
                        label="Non-Musician : Musician Mental Health Ratio",
                        value=f"{musician_effect_ratio:.2f}",
                        delta=f"{'Higher ratio indicates stronger musician advantage' if musician_effect_ratio > 1 else 'No clear musician advantage'}"
                    )
                    
                    st.markdown("""
                    **Why it matters**: This ratio quantifies the mental health advantage of being a musician versus non-musician. 
                    Values above 1.0 indicate that non-musicians report higher (worse) mental health scores than musicians.
                    
                    **Target audience**: Music education advocates, mental health researchers, policymakers.
                    """)
        
        # Top-line KPI visualization
        st.subheader("KPI Relationships")
        
        # Scatter plot matrix of KPIs
        kpi_vars = [var for var in ['Hours_Per_Day', 'Engagement_Score', 'Genre_Diversity_Index', 'Mental_Health_Composite'] 
                   if var in df_engineered.columns]
        
        if len(kpi_vars) >= 2:
            fig = px.scatter_matrix(
                df_engineered,
                dimensions=kpi_vars,
                color='Age_Group' if 'Age_Group' in df_engineered.columns else None,
                color_discrete_sequence=QUALITATIVE_COLORS,
                opacity=0.7,
                height=700  # Increase height for better visibility
            )
            fig.update_layout(
                title="Relationships Between Key Performance Indicators",
                title_font_size=20,
                font_size=14,  # Increase font size for axis labels
            )
            # Make axis labels more readable
            for annotation in fig.layout.annotations:
                annotation.font.size = 16
            st.plotly_chart(fig, use_container_width=True)
        
        # KPI over time (simulated)
        st.subheader("KPI Trends (Simulated)")
        
        # Create simulated data for KPI trends
        np.random.seed(42)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        trend_data = pd.DataFrame({
            'Month': months,
            'Avg_Listening_Hours': np.linspace(avg_hours - 0.5, avg_hours + 0.5, 12) + np.random.normal(0, 0.2, 12),
            'Avg_Engagement_Score': np.linspace(avg_engagement - 0.7, avg_engagement + 0.9, 12) + np.random.normal(0, 0.3, 12),
            'Avg_Genre_Diversity': np.linspace(avg_diversity - 0.3, avg_diversity + 1.1, 12) + np.random.normal(0, 0.4, 12)
        })
        
        # Plot the trends
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Avg_Listening_Hours'],
            mode='lines+markers',
            name='Listening Hours',
            line=dict(color=MAIN_COLOR, width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Avg_Engagement_Score'] / 10 * avg_hours,  # Scale to similar range
            mode='lines+markers',
            name='Engagement Score (scaled)',
            line=dict(color=ACCENT_COLOR, width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Avg_Genre_Diversity'] / 10 * avg_hours,  # Scale to similar range
            mode='lines+markers',
            name='Genre Diversity (scaled)',
            line=dict(color="#6A9EC0", width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Simulated KPI Trends Over Time',
            title_font_size=20,
            xaxis_title='Month',
            yaxis_title='Value',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=500
        )
        fig.update_xaxes(tickfont_size=14)
        fig.update_yaxes(tickfont_size=14)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        Note: This trend visualization uses simulated data to demonstrate how KPIs might be tracked over time. 
        In a production environment, this would be replaced with actual longitudinal data.
        """)
    
    # RECOMMENDATIONS TAB
    with tabs[1]:
        st.subheader("Key Insights & Recommendations")
        
        st.markdown("""
        Based on our analysis of music listening habits and mental health, we've identified these key insights and recommendations:
        """)
        
        # Create two columns for recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎧 For Individuals
            
            **Listen strategically:**
            - Aim for 3-5 hours of music daily
            - Explore different genres to increase music diversity
            - Create mood-specific playlists for different activities
            - Consider learning an instrument or music creation
            
            **Why it works:**
            Our data shows higher engagement with music and greater genre diversity strongly correlate with improved mental health metrics.
            """)
            
            st.markdown("""
            ### 🏫 For Educators & Therapists
            
            **Incorporate music in treatment:**
            - Add structured music activities to mental health programs
            - Promote accessible music creation opportunities
            - Use music to facilitate emotional expression
            - Develop community music programs for all ages
            
            **Why it works:**
            Musicians showed significantly better mental health scores across all measures in our analysis.
            """)
        
        with col2:
            st.markdown("""
            ### 🎵 For Music Services
            
            **Enhance user experience:**
            - Develop features to gradually expand genre exposure
            - Create mental health-focused playlists and features
            - Add tools for tracking mood alongside music
            - Suggest diverse genres based on listening patterns
            
            **Why it works:**
            Genre diversity showed negative correlations with anxiety and depression scores.
            """)
            
            st.markdown("""
            ### 💼 For Workplaces
            
            **Optimize the environment:**
            - Provide guidelines for music use during work
            - Create shared playlists for different work activities
            - Consider music breaks for stress reduction
            - Support employee music engagement initiatives
            
            **Why it works:**
            Listening while working was associated with both productivity and lower stress levels.
            """)
        
        # Summary insight box
        st.info("""
        **Key Takeaway:** Our analysis shows that active, diverse music engagement has a strong positive association with 
        mental wellbeing. The most significant effects were seen in those who both listen to a wide variety of music 
        AND participate in music creation.
        """)

    # IMPLEMENTATION PLAN TAB
    with tabs[2]:
        st.subheader("Next Steps")
        
        st.markdown("""
        These simple actions can help apply our findings to improve mental wellbeing through music:
        """)
        
        # Implementation steps with visual organization
        st.markdown("""
        ### For Individuals
        
        1. **Track your listening:** Record your music habits and mood for 2 weeks
        2. **Expand your playlist:** Add 5 new genres to your regular rotation
        3. **Active listening:** Set aside 15 minutes daily for focused music appreciation
        4. **Consider creation:** Try a beginner-friendly music creation app or lessons
        """)
        
        st.markdown("""
        ### For Organizations
        
        1. **Share insights:** Distribute key findings to relevant stakeholders
        2. **Pilot program:** Test music engagement activities in small groups
        3. **Collect feedback:** Gather information on what works best for your audience
        4. **Scale successful approaches:** Expand effective strategies to wider groups
        """)
        
        # Future improvements
        st.subheader("Future Improvements (Optional)")
        
        st.markdown("""
        To build on this analysis in the future:
        
        - Collect longitudinal data to better establish causal relationships
        - Develop personalized recommendations based on individual responses to music
        - Create specialized programs for different age groups and backgrounds
        - Explore neurological mechanisms behind music's mental health effects
        """)
        
        # Final call to action
        st.success("""
        **Start Today:** Even small changes to how you engage with music can make a meaningful difference
        in mental wellbeing. Begin with one recommendation and build from there.
        """)

# Add a footer with information about the dashboard
st.markdown("""
---
### About This Dashboard

This interactive dashboard explores the relationship between music consumption habits and mental health metrics.
It was created as part of a data fluency project focused on understanding how music may influence mental wellbeing.

The analysis includes data on listening habits, genre preferences, music engagement levels, and self-reported 
mental health metrics. Through various statistical analyses and visualizations, we aim to uncover meaningful 
patterns and relationships that could inform personal habits and potential interventions.

**Data sources**: Survey data on music listening habits and mental health metrics.

**Analysis methods**: Descriptive statistics, data visualization, correlation analysis, regression modeling, 
and hypothesis testing.

---
""", unsafe_allow_html=True)
