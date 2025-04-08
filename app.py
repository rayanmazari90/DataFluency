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

# Apply custom styling for dark theme
st.markdown("""
<style>
    /* Force dark theme */
    html, body, [class*="css"] {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }
    .main {
        padding: 1rem;
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #073763;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
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
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 8px 16px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B6AA0;
        color: white;
    }
    /* Increase plot height */
    .stPlotlyChart {
        min-height: 500px;
    }
    /* Force dark theme elements */
    div.stDataFrame {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    div.stTable {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .css-1d391kg, .css-14xtw13 {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .streamlit-expanderHeader {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .stSelectbox {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .streamlit-expanderContent {
        background-color: #0E1117;
        color: #FAFAFA;
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
    
    # Ensure boolean columns are properly converted to boolean values first, then to float
    bool_columns = ['While_Working', 'Instrumentalist', 'Composer', 'Foreign_Languages']
    for col in bool_columns:
        if col in df_engineered.columns:
            # Handle different possible values for boolean columns
            if df_engineered[col].dtype == 'object':
                df_engineered[col] = df_engineered[col].map(lambda x: x.lower() in ('yes', 'y', 'true', '1') if isinstance(x, str) else bool(x))
            # Now it should be safe to convert to float
            df_engineered[col] = df_engineered[col].astype(float)
    
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
        df_engineered['Engagement_Score'] += df_engineered['While_Working'] * 2
    
    # Add points for music creation (instrumentalist/composer)
    if 'Instrumentalist' in df_engineered.columns:
        df_engineered['Engagement_Score'] += df_engineered['Instrumentalist'] * 3
    
    if 'Composer' in df_engineered.columns:
        df_engineered['Engagement_Score'] += df_engineered['Composer'] * 4
    
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
        foreign_lang_value = df_engineered['Foreign_Languages']
    
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
    
    # Add Team and QR code at the beginning
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Team Members
        """)
        
        # Create a 3-column layout for team members, with consistent image size
        image_width = 120  # Set consistent width for all images
        team_col1, team_col2, team_col3 = st.columns(3)
        
        # Custom function to display team member image with consistent sizing
        def display_team_member(image_path, name, width=image_width):
            # Apply CSS for consistent sizing and circular cropping
            st.markdown(f"""
            <style>
            .team-img-{name.replace(" ", "-")} {{
                width: {width}px;
                height: {width}px;
                object-fit: cover;
                border-radius: 50%;
                margin-bottom: 10px;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            # Display the image with HTML to apply the custom styling
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{get_image_base64(image_path)}" class="team-img-{name.replace(" ", "-")}" alt="{name}">
                <p><strong>{name}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # Function to convert image to base64
        def get_image_base64(image_path):
            import base64
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')

        with team_col1:
            display_team_member("pictures/Rayane Boumediene Mazari.jpeg", "Rayane Boumediene Mazari")
            display_team_member("pictures/Jacob.jpeg", "Jacob")

        with team_col2:
            display_team_member("pictures/Alp Eyupoglu.jpeg", "Alp Eyupoglu")
            display_team_member("pictures/Matteo.jpeg", "Matteo")

        with team_col3:
            display_team_member("pictures/Maria Alcalde.jpeg", "Maria Alcalde")
            display_team_member("pictures/José ,aria Teixeira.jpeg", "José Maria Teixeira")
    
    with col2:
        st.markdown("""
        ### Try Our Live Dashboard
        Scan this QR code to access the dashboard:
        """)
        st.image("https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://datafluencygroup5.streamlit.app", width=200)
        st.markdown("[https://datafluencygroup5.streamlit.app](https://datafluencygroup5.streamlit.app)")
    
    st.markdown("---")
    
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
    
    # Add explanation box about mental health metrics interpretation
    st.info("""
    **Important Note on Mental Health Metrics**: For all mental health metrics (Anxiety, Depression, Insomnia, and OCD), 
    **lower scores indicate better mental health** (fewer symptoms). Scores range from 0-10, where 0 represents 
    no symptoms and 10 represents maximum severity of symptoms.
    """)
    
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
        # Sort descending for horizontal bars - most popular genres at the top
        favorite_genre_counts = favorite_genre_counts.sort_values('Count', ascending=False)

        # Create horizontal bar chart
        fig = px.bar(favorite_genre_counts, 
                     y='Genre',  # Genre on y-axis for horizontal bars
                     x='Count',  # Count on x-axis
                     title='Distribution of Favorite Genres',
                     color='Genre', 
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     orientation='h')  # Explicitly set orientation to horizontal

        # Adjust layout for better readability
        fig.update_layout(
            yaxis_title='Genre', 
            xaxis_title='Count',
            height=500,  # Increase height to accommodate all genres
            margin=dict(l=150),  # Add left margin for genre labels
            yaxis=dict(automargin=True)  # Auto-adjust margins for y-axis labels
        )
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
        
        st.header("Correlation Between Music Consumption and Mental Health")
        
        # Create correlation matrix visualization
        st.subheader("Correlation Matrix between Music Variables and Mental Health")
        
        # Calculate the correlation matrix using df_engineered
        # Extract music variables and mental health variables
        music_vars = ['Hours_Per_Day', 'Genre_Diversity_Index', 'Engagement_Score', 'Music_Versatility', 'While_Working']
        mental_health_vars = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
        
        # Extract the relevant subset of the correlation matrix
        music_mental_corr = df_engineered[music_vars + mental_health_vars].corr().loc[music_vars, mental_health_vars]
        
        # Create the focused heatmap with sequential color scale
        fig = px.imshow(
            music_mental_corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="OrRd",  # Sequential color scale
            title="Music Variables vs Mental Health Correlations",
            labels=dict(x="Mental Health Metrics", y="Music Variables", color="Correlation")
        )
        
        fig.update_layout(
            width=800,
            height=500,
            xaxis_title="Mental Health Variables",
            yaxis_title="Music Variables"
        )
        
        st.plotly_chart(fig)
        
        # Provide insights based on the correlation matrix
        st.markdown("""
        **Key Insights from the Correlation Analysis:**
        
        1. **Weak Positive Correlations**: The data shows mostly weak positive correlations between music variables and mental health symptoms. This suggests that as music engagement increases, there may be a slight tendency toward higher symptom scores, though these relationships are not strong.
        
        2. **Hours_Per_Day**: Time spent listening to music shows weak positive correlations with all mental health measures (values ranging from ~0.05 to ~0.14), with the strongest correlation with Insomnia (0.14).
        
        3. **Genre_Diversity_Index**: Diversity in music genres shows weak positive correlations with mental health symptoms, particularly with Depression (0.17).
        
        4. **Music_Versatility**: The variety of contexts in which people listen to music has the strongest correlation with Anxiety (0.11) compared to other symptoms.
        
        5. **Engagement_Score**: Overall music engagement shows weak positive correlations across all mental health metrics (0.05-0.15).
        
        **Interpretation Note**: These weak positive correlations could suggest that people experiencing mental health symptoms might use music more as a coping mechanism rather than music causing these symptoms. The correlation analysis alone cannot establish causation or direction of influence.
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
        st.subheader("Advanced Musical Engagement Comparison")

        # Create a categorical variable with three distinct groups
        df_engineered['Musical_Status'] = 'Non-Musical'
        df_engineered.loc[(df_engineered['Instrumentalist'] == True) & (df_engineered['Composer'] == False), 'Musical_Status'] = 'Instrumentalist Only'
        df_engineered.loc[(df_engineered['Instrumentalist'] == False) & (df_engineered['Composer'] == True), 'Musical_Status'] = 'Composer Only'
        df_engineered.loc[(df_engineered['Instrumentalist'] == True) & (df_engineered['Composer'] == True), 'Musical_Status'] = 'Both Instrumentalist & Composer'

        # Define mental health metrics
        mental_health_metrics = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

        # Prepare data for grouped bar chart with the new categories
        musician_mh = df_engineered.groupby('Musical_Status')[mental_health_metrics].mean().reset_index()
        musician_mh_melted = pd.melt(musician_mh, id_vars=['Musical_Status'], 
                                    value_vars=mental_health_metrics,
                                    var_name='Metric', value_name='Score')

        # Create a more informative grouped bar chart
        fig = px.bar(musician_mh_melted, x='Metric', y='Score', color='Musical_Status',
                    barmode='group', 
                    title='Mental Health Metrics by Musical Engagement Type',
                    color_discrete_sequence=['#F08080', '#6495ED', '#90EE90', '#FFD700'])

        # Add horizontal line at the overall average for reference
        overall_avg = df_engineered[mental_health_metrics].mean().mean()
        fig.add_shape(
            type="line",
            x0=-0.5, y0=overall_avg,
            x1=3.5, y1=overall_avg,
            line=dict(color="black", width=2, dash="dash"),
        )

        # Add annotation for the reference line
        fig.add_annotation(
            x=3.2, y=overall_avg + 0.2,
            text=f"Overall Average: {overall_avg:.2f}",
            showarrow=False,
            font=dict(size=10)
        )
        
        # Add note about mental health metrics interpretation
        fig.update_layout(
            xaxis_title='Mental Health Metric', 
            yaxis_title='Average Score (Lower is Better)',
            annotations=[
                dict(
                    x=0.5, y=-0.15,
                    xref="paper", yref="paper",
                    #text="Note: Lower scores indicate fewer symptoms (better mental health)",
                    showarrow=False,
                    font=dict(size=12, color="red")
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add detailed explanation of the comparison
        st.info("""
        **Mental Health Metrics Interpretation:**
        - Lower scores indicate fewer symptoms (better mental health)
        - Higher scores indicate more symptoms (worse mental health)
        - The scores range from 0-10, with 0 being no symptoms and 10 being severe symptoms
        """)

        # Add statistical analysis of the differences
        st.subheader("Statistical Analysis of Group Differences")

        # Calculate the average mental health composite for each group
        group_means = df_engineered.groupby('Musical_Status')['Mental_Health_Composite'].mean().reset_index()
        group_means = group_means.sort_values('Mental_Health_Composite')  # Sort by mental health score

        # Create horizontal bar chart for overall mental health comparison
        fig = px.bar(
            group_means, 
            y='Musical_Status', 
            x='Mental_Health_Composite',
            title='Overall Mental Health Composite by Musical Engagement Type',
            color='Musical_Status',
            color_discrete_sequence=['#90EE90', '#6495ED', '#FFD700', '#F08080'],
            orientation='h'
        )

        fig.update_layout(
            xaxis_title='Mental Health Composite Score (Lower is Better)',
            yaxis_title='Musical Status',
            yaxis=dict(autorange="reversed")  # Reverse y-axis to match order of table
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show insight from the more detailed analysis
        st.markdown("""
        **Insights from Advanced Comparison:**
        
        1. **Engagement Level Impact**: People who are both instrumentalists and composers show the highest mental health symptom scores across all metrics, which suggests that deeper engagement with music creation may be associated with higher reported symptoms of anxiety, depression, insomnia, and OCD.
        
        2. **Different Types of Engagement**: There are noticeable differences between those who only play instruments versus those who only compose music, with composers generally reporting higher mental health symptom scores than instrumentalists only.
        
        3. **Gradient Effect**: The data suggests a pattern where those more engaged in music creation (composers and those who are both composers and instrumentalists) report higher levels of mental health symptoms than non-musicians or instrumentalists only.
        
        4. **Specific Symptom Patterns**: Anxiety and depression scores show the largest differences between groups, with anxiety scores being particularly elevated in the "Both Instrumentalist & Composer" group.
        
        **Note on Interpretation**: It's important to note that this finding contradicts some existing research on music and mental health. This could suggest that people with existing mental health conditions might be more drawn to musical creation as a coping mechanism, rather than music causing mental health symptoms. Alternatively, it might reflect aspects of the professional music industry that can be stressful and demanding.
        
        These findings highlight the complex relationship between musical engagement and mental health that requires careful interpretation.
        """)

elif page == "Hypothesis Testing":
    st.title("🧪 In-Depth Analysis & Optimal Ranges")
    
    st.markdown("""
    This section explores how different levels of our key music metrics relate to specific mental health outcomes, 
    helping us identify optimal ranges for each metric.
    """)
    
    # Create tabs for different music metrics
    tabs = st.tabs(["Genre Diversity", "Engagement Score", "Music Versatility", "Optimal Listening Hours"])
    
    # GENRE DIVERSITY TAB
    with tabs[0]:
        st.subheader("Genre Diversity Index: Impact on Mental Health")
        
        st.markdown("""
        How does listening to diverse genres of music relate to mental health outcomes? 
        Here we explore the relationship between Genre Diversity Index and specific mental health metrics.
        """)
        
        # Create categorical variable for Genre Diversity
        diversity_bins = [0, 2, 4, 6, 8, 10]
        diversity_labels = ['Very Low (0-2)', 'Low (2-4)', 'Medium (4-6)', 'High (6-8)', 'Very High (8-10)']
        df_engineered['Diversity_Category'] = pd.cut(df_engineered['Genre_Diversity_Index'], 
                                                  bins=diversity_bins, labels=diversity_labels)
        
        # Analysis by mental health metric
        mental_health_metrics = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
        
        # Allow user to select specific mental health metric
        selected_mh = st.selectbox('Select mental health metric:', 
                                  mental_health_metrics + ['Mental_Health_Composite'],
                                  index=4)
        
        # Calculate stats by diversity level
        diversity_stats = df_engineered.groupby('Diversity_Category')[selected_mh].agg(['mean', 'std', 'count']).reset_index()
        diversity_stats.columns = ['Diversity_Level', 'Mean', 'Std', 'Count']
        diversity_stats['SE'] = diversity_stats['Std'] / np.sqrt(diversity_stats['Count'])
        
        # Find optimal diversity level
        optimal_level_idx = diversity_stats['Mean'].idxmin()
        optimal_diversity = diversity_stats.iloc[optimal_level_idx]['Diversity_Level']
        
        # Create bar chart with error bars
        fig = px.bar(diversity_stats, x='Diversity_Level', y='Mean',
                   error_y='SE',
                   title=f'{selected_mh} by Genre Diversity Level',
                   color='Mean', 
                   color_continuous_scale='RdBu_r',
                   labels={'Mean': f'Mean {selected_mh} Score', 'Diversity_Level': 'Genre Diversity Level'})
        
        # Highlight optimal diversity level
        fig.add_annotation(
            x=optimal_diversity,
            y=diversity_stats.iloc[optimal_level_idx]['Mean'],
            text=f"Optimal: {optimal_diversity}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        fig.update_layout(
            xaxis_title='Genre Diversity Level',
            yaxis_title=f'{selected_mh} Score (Lower is Better)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show breakdown for all mental health metrics
        st.subheader("Detailed Breakdown by Mental Health Metric")
        
        # Prepare data for all metrics
        all_metrics_data = []
        
        for metric in mental_health_metrics:
            metric_data = df_engineered.groupby('Diversity_Category')[metric].mean().reset_index()
            metric_data['Metric'] = metric
            metric_data.columns = ['Diversity_Level', 'Score', 'Metric']
            all_metrics_data.append(metric_data)
        
        all_metrics_df = pd.concat(all_metrics_data)
        
        # Create grouped bar chart
        fig = px.bar(all_metrics_df, x='Diversity_Level', y='Score', color='Metric',
                   barmode='group',
                   title='Mental Health Metrics by Genre Diversity Level',
                   labels={'Score': 'Mental Health Score', 'Diversity_Level': 'Genre Diversity Level'})
        
        fig.update_layout(
            xaxis_title='Genre Diversity Level',
            yaxis_title='Mental Health Score (Lower is Better)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate recommended range for each mental health metric
        st.subheader("Recommended Genre Diversity Ranges")
        
        optimal_ranges = {}
        
        for metric in mental_health_metrics + ['Mental_Health_Composite']:
            metric_by_diversity = df_engineered.groupby('Diversity_Category')[metric].mean()
            if not metric_by_diversity.empty:
                # Find the diversity category with minimum symptom score
                optimal_idx = metric_by_diversity.argmin()
                optimal_ranges[metric] = metric_by_diversity.index[optimal_idx]
            else:
                optimal_ranges[metric] = "N/A"
        
        # Create recommendation table
        recommendations = pd.DataFrame({
            'Mental Health Metric': list(optimal_ranges.keys()),
            'Optimal Genre Diversity Range': list(optimal_ranges.values())
        })
        
        # Display without problematic styling
        st.dataframe(recommendations)
        
        # Key insights
        st.info("""
        **Key Insights on Genre Diversity**:
        
        1. **Medium to High diversity** (4-8 range) generally shows the best mental health outcomes across all metrics.
        
        2. **Different mental health aspects** respond slightly differently to genre diversity, 
           with anxiety and depression showing the strongest relationship with genre diversity.
        
        3. **Very Low or Very High diversity** tend to be associated with higher symptom scores, 
           suggesting moderation is important.
        
        This evidence supports our KPI recommendation for a genre diversity target range 
        of 4-8 on our 10-point scale.
        """)
    
    # ENGAGEMENT SCORE TAB
    with tabs[1]:
        st.subheader("Music Engagement Score: Impact on Mental Health")
        
        st.markdown("""
        How does the level of engagement with music (listening and creation) relate to mental health outcomes?
        This analysis explores the relationship between Engagement Score and specific mental health metrics.
        """)
        
        # Create categorical variable for Engagement Score
        engagement_bins = [0, 2, 4, 6, 8, 10]
        engagement_labels = ['Very Low (0-2)', 'Low (2-4)', 'Medium (4-6)', 'High (6-8)', 'Very High (8-10)']
        df_engineered['Engagement_Category'] = pd.cut(df_engineered['Engagement_Score'], 
                                                   bins=engagement_bins, labels=engagement_labels)
        
        # Allow user to select specific mental health metric
        selected_mh = st.selectbox('Select mental health metric:', 
                                  mental_health_metrics + ['Mental_Health_Composite'],
                                  index=4,
                                  key='engagement_mh_select')
        
        # Calculate stats by engagement level
        engagement_stats = df_engineered.groupby('Engagement_Category')[selected_mh].agg(['mean', 'std', 'count']).reset_index()
        engagement_stats.columns = ['Engagement_Level', 'Mean', 'Std', 'Count']
        engagement_stats['SE'] = engagement_stats['Std'] / np.sqrt(engagement_stats['Count'])
        
        # Find optimal engagement level
        optimal_level_idx = engagement_stats['Mean'].idxmin()
        optimal_engagement = engagement_stats.iloc[optimal_level_idx]['Engagement_Level']
        
        # Create bar chart with error bars
        fig = px.bar(engagement_stats, x='Engagement_Level', y='Mean',
                   error_y='SE',
                   title=f'{selected_mh} by Music Engagement Level',
                   color='Mean', 
                   color_continuous_scale='RdBu_r',
                   labels={'Mean': f'Mean {selected_mh} Score', 'Engagement_Level': 'Music Engagement Level'})
        
        # Highlight optimal engagement level
        fig.add_annotation(
            x=optimal_engagement,
            y=engagement_stats.iloc[optimal_level_idx]['Mean'],
            text=f"Optimal: {optimal_engagement}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        fig.update_layout(
            xaxis_title='Music Engagement Level',
            yaxis_title=f'{selected_mh} Score (Lower is Better)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Musician vs Non-musician comparison
        st.subheader("Music Engagement Type Analysis")
        
        # Create a categorical variable with musician groups
        if 'Musical_Status' not in df_engineered.columns:
            df_engineered['Musical_Status'] = 'Non-Musical'
            df_engineered.loc[(df_engineered['Instrumentalist'] == True) & (df_engineered['Composer'] == False), 'Musical_Status'] = 'Instrumentalist Only'
            df_engineered.loc[(df_engineered['Instrumentalist'] == False) & (df_engineered['Composer'] == True), 'Musical_Status'] = 'Composer Only'
            df_engineered.loc[(df_engineered['Instrumentalist'] == True) & (df_engineered['Composer'] == True), 'Musical_Status'] = 'Both Instrumentalist & Composer'
        
        # Prepare data for grouped bar chart
        musician_mh = df_engineered.groupby('Musical_Status')[mental_health_metrics].mean().reset_index()
        musician_mh_melted = pd.melt(musician_mh, id_vars=['Musical_Status'], 
                                    value_vars=mental_health_metrics,
                                    var_name='Metric', value_name='Score')
        
        # Create grouped bar chart
        fig = px.bar(musician_mh_melted, x='Metric', y='Score', color='Musical_Status',
                   barmode='group',
                   title='Mental Health Metrics by Musical Engagement Type',
                   color_discrete_sequence=['#F08080', '#6495ED', '#90EE90', '#FFD700'])
        
        # Add reference line for overall average
        overall_avg = df_engineered[mental_health_metrics].mean().mean()
        fig.add_shape(
            type="line",
            x0=-0.5, y0=overall_avg,
            x1=3.5, y1=overall_avg,
            line=dict(color="black", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=3.2, y=overall_avg + 0.2,
            text=f"Overall Average: {overall_avg:.2f}",
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.update_layout(
            xaxis_title='Mental Health Metric',
            yaxis_title='Average Score (Lower is Better)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate recommended range for each mental health metric
        st.subheader("Recommended Engagement Score Ranges")
        
        optimal_ranges = {}
        
        for metric in mental_health_metrics + ['Mental_Health_Composite']:
            metric_by_engagement = df_engineered.groupby('Engagement_Category')[metric].mean()
            if not metric_by_engagement.empty:
                # Find the engagement category with minimum symptom score
                optimal_idx = metric_by_engagement.argmin()
                optimal_ranges[metric] = metric_by_engagement.index[optimal_idx]
            else:
                optimal_ranges[metric] = "N/A"
        
        # Create recommendation table
        recommendations = pd.DataFrame({
            'Mental Health Metric': list(optimal_ranges.keys()),
            'Optimal Engagement Range': list(optimal_ranges.values())
        })
        
        # Display without problematic styling
        st.dataframe(recommendations)
        
        # Key insights
        st.info("""
        **Key Insights on Music Engagement**:
        
        1. **Moderate levels of engagement** are generally associated with better mental health outcomes, 
           but the pattern shows an interesting U-shape where very low and very high engagement relate to higher symptom scores.
        
        2. **Type of engagement matters**: People who are both instrumentalists and composers show 
           higher mental health symptom scores across all metrics compared to other groups.
        
        3. **Anxiety and Depression** show the strongest differences between musician groups, 
           with anxiety scores being particularly elevated in the "Both Instrumentalist & Composer" group.
        
        These findings suggest a more nuanced view of music engagement than "more is better" - 
        moderate engagement appears optimal, and highly engaged musicians may need additional support.
        """)
    
    # MUSIC VERSATILITY TAB  
    with tabs[2]:
        st.subheader("Music Versatility: Impact on Mental Health")
        
        st.markdown("""
        How does versatility in music listening (combining genre diversity, exploratory behavior, and 
        foreign language music) relate to mental health outcomes?
        """)
        
        # Create categorical variable for Music Versatility
        versatility_bins = [0, 2, 4, 6, 8, 10]
        versatility_labels = ['Very Low (0-2)', 'Low (2-4)', 'Medium (4-6)', 'High (6-8)', 'Very High (8-10)']
        df_engineered['Versatility_Category'] = pd.cut(df_engineered['Music_Versatility'], 
                                                    bins=versatility_bins, labels=versatility_labels)
        
        # Allow user to select specific mental health metric
        selected_mh = st.selectbox('Select mental health metric:', 
                                  mental_health_metrics + ['Mental_Health_Composite'],
                                  index=4,
                                  key='versatility_mh_select')
        
        # Calculate stats by versatility level
        versatility_stats = df_engineered.groupby('Versatility_Category')[selected_mh].agg(['mean', 'std', 'count']).reset_index()
        versatility_stats.columns = ['Versatility_Level', 'Mean', 'Std', 'Count']
        versatility_stats['SE'] = versatility_stats['Std'] / np.sqrt(versatility_stats['Count'])
        
        # Find optimal versatility level
        optimal_level_idx = versatility_stats['Mean'].idxmin()
        if optimal_level_idx < len(versatility_stats):
            optimal_versatility = versatility_stats.iloc[optimal_level_idx]['Versatility_Level']
        else:
            optimal_versatility = "N/A"
        
        # Create bar chart with error bars
        fig = px.bar(versatility_stats, x='Versatility_Level', y='Mean',
                   error_y='SE',
                   title=f'{selected_mh} by Music Versatility Level',
                   color='Mean', 
                   color_continuous_scale='RdBu_r',
                   labels={'Mean': f'Mean {selected_mh} Score', 'Versatility_Level': 'Music Versatility Level'})
        
        # Highlight optimal versatility level if it exists
        if optimal_versatility != "N/A":
            fig.add_annotation(
                x=optimal_versatility,
                y=versatility_stats.iloc[optimal_level_idx]['Mean'],
                text=f"Optimal: {optimal_versatility}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        
        fig.update_layout(
            xaxis_title='Music Versatility Level',
            yaxis_title=f'{selected_mh} Score (Lower is Better)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Components of music versatility
        st.subheader("Components of Music Versatility")
        
        # Create a more informative visualization instead of the scatter plot
        # Use a heatmap to show the relationship between components
        versatility_corr = df_engineered[['Genre_Diversity_Index', 'Exploratory_Level', 'Foreign_Languages', 'Music_Versatility', selected_mh]].corr()
        
        # Create a heatmap
        fig = px.imshow(
            versatility_corr,
            color_continuous_scale='RdBu_r',
            title=f'Correlation Between Music Versatility Components and {selected_mh}',
            labels=dict(x='Component', y='Component', color='Correlation'),
            text_auto=True,
            aspect="auto"
        )
        
        fig.update_layout(
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a more informative radar chart to show component contributions
        # Select a few representative rows with different versatility levels
        if len(df_engineered) > 10:
            versatility_levels = df_engineered['Versatility_Category'].unique()
            sample_data = []
            
            for level in versatility_levels:
                # Get one representative sample from each level
                level_data = df_engineered[df_engineered['Versatility_Category'] == level]
                if not level_data.empty:
                    # Find the row with the lowest mental health score in this level
                    best_idx = level_data[selected_mh].idxmin()
                    sample_data.append(level_data.loc[best_idx])
            
            if sample_data:
                # Create a DataFrame with the samples
                samples_df = pd.DataFrame(sample_data)
                
                # Create separate radar charts for each versatility level to avoid length mismatch
                st.subheader("Music Versatility Profiles by Category")
                
                # Create a multi-line radar chart using go.Figure instead of px.line_polar
                radar_vars = ['Genre_Diversity_Index', 'Exploratory_Level', 'Foreign_Languages', selected_mh]
                
                fig = go.Figure()
                
                colors = px.colors.qualitative.Bold
                for i, row in samples_df.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row[var] for var in radar_vars],
                        theta=radar_vars,
                        fill='toself',
                        name=f"{row['Versatility_Category']}",
                        line_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    title=f'Music Versatility Components Profile by Category (Lower {selected_mh} is Better)',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a table with the component values for each versatility level
                st.subheader("Detailed Component Values")
                display_cols = ['Versatility_Category'] + radar_vars
                st.dataframe(samples_df[display_cols].set_index('Versatility_Category'))
        
        # Calculate recommended range for each mental health metric
        st.subheader("Recommended Music Versatility Ranges")
        
        optimal_ranges = {}
        
        for metric in mental_health_metrics + ['Mental_Health_Composite']:
            metric_by_versatility = df_engineered.groupby('Versatility_Category')[metric].mean()
            if not metric_by_versatility.empty:
                # Find the versatility category with minimum symptom score
                optimal_idx = metric_by_versatility.argmin()
                optimal_ranges[metric] = metric_by_versatility.index[optimal_idx]
            else:
                optimal_ranges[metric] = "N/A"
        
        # Create recommendation table
        recommendations = pd.DataFrame({
            'Mental Health Metric': list(optimal_ranges.keys()),
            'Optimal Versatility Range': list(optimal_ranges.values())
        })
        
        # Display without problematic styling
        st.dataframe(recommendations)
        
        # Key insights
        st.info("""
        **Key Insights on Music Versatility**:
        
        1. **Medium to High versatility** in music consumption (combining diverse genres, exploratory listening, 
           and foreign language music) generally shows better mental health outcomes.
        
        2. **Foreign language music** appears to have a particularly interesting relationship with anxiety levels, 
           potentially offering beneficial effects.
        
        3. **Versatility components interact**: The most beneficial pattern combines moderate-to-high genre diversity 
           with medium exploratory listening behaviors.
        
        These findings suggest that encouraging versatile but balanced music consumption patterns 
        could be valuable for mental health interventions.
        """)
    
    # OPTIMAL LISTENING HOURS TAB
    with tabs[3]:
        st.subheader("Optimal Listening Hours Analysis")
        
        st.markdown("""
        What is the ideal amount of time to spend listening to music each day for optimal mental health?
        This analysis identifies the "sweet spot" for music listening duration.
        """)
        
        # Create categorical variable for hours
        hour_bins = [0, 1, 2, 3, 4, 5, 6, 7, 10]
        hour_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-10']
        df_engineered['Hours_Category'] = pd.cut(df_engineered['Hours_Per_Day'], bins=hour_bins, labels=hour_labels)
        
        # Allow user to select specific mental health metric
        selected_mh = st.selectbox('Select mental health metric:', 
                                  mental_health_metrics + ['Mental_Health_Composite'],
                                  index=4,
                                  key='hours_mh_select')
        
        # Calculate stats by hours category
        hours_stats = df_engineered.groupby('Hours_Category')[selected_mh].agg(['mean', 'std', 'count']).reset_index()
        hours_stats.columns = ['Hours_Range', 'Mean', 'Std', 'Count']
        hours_stats['SE'] = hours_stats['Std'] / np.sqrt(hours_stats['Count'])
        
        # Find optimal hours range
        optimal_hours_idx = hours_stats['Mean'].idxmin()
        optimal_hours = hours_stats.iloc[optimal_hours_idx]['Hours_Range']
        
        # Create bar chart with error bars
        fig = px.bar(hours_stats, x='Hours_Range', y='Mean',
                   error_y='SE',
                   title=f'{selected_mh} by Daily Listening Hours',
                   color='Mean', 
                   color_continuous_scale='RdBu_r',
                   labels={'Mean': f'Mean {selected_mh} Score', 'Hours_Range': 'Hours Per Day'})
        
        # Highlight optimal hours range
        fig.add_annotation(
            x=optimal_hours,
            y=hours_stats.iloc[optimal_hours_idx]['Mean'],
            text=f"Optimal: {optimal_hours}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        fig.update_layout(
            xaxis_title='Hours of Music Per Day',
            yaxis_title=f'{selected_mh} Score (Lower is Better)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        

        # Calculate recommended hours for each mental health metric
        st.subheader("Recommended Listening Hours by Mental Health Metric")
        
        optimal_ranges = {}
        
        for metric in mental_health_metrics + ['Mental_Health_Composite']:
            metric_by_hours = df_engineered.groupby('Hours_Category')[metric].mean()
            if not metric_by_hours.empty:
                # Find the hour category with minimum symptom score
                optimal_idx = metric_by_hours.argmin()
                optimal_ranges[metric] = metric_by_hours.index[optimal_idx]
            else:
                optimal_ranges[metric] = "N/A"
            
        # Display the results
        recommendations = pd.DataFrame({
            'Mental Health Metric': list(optimal_ranges.keys()),
            'Optimal Listening Hours': list(optimal_ranges.values())
        })
        
        # Display without problematic styling
        st.dataframe(recommendations)
        
        # Key insights - Updated to reflect the actual data findings
        st.info("""
        **Key Insights on Listening Hours**:
        
        1. **0-1 hours per day** consistently shows the best mental health outcomes across most metrics, 
           suggesting this is the "sweet spot" for music listening.
        
        2. **U-shaped relationship**: Both too little and too much music consumption relate to higher 
           symptom scores, confirming that moderation is key.
        
        3. **Different metrics, similar pattern**: The data consistently shows that shorter listening durations 
           (0-1 hours) are associated with the lowest symptom scores across mental health metrics.
        
        This evidence supports our KPI recommendation of 0-1 hours of daily music listening 
        as the optimal target range.
        """)
    
    # SUMMARY SECTION
    st.subheader("Summary of Optimal Ranges Analysis")
    
    st.markdown("""
    Our in-depth analysis of key music metrics and their relationship with mental health outcomes reveals 
    several actionable insights:
    """)
    
    # Create summary table of all optimal ranges
    optimal_summary = {
        'Metric': ['Genre Diversity Index', 'Engagement Score', 'Music Versatility', 'Listening Hours'],
        'Optimal Range': ['Medium to High (4-8)', 'Medium (4-6)', 'Medium to High (4-8)', '0-1 hours/day'],
        'Impact on Mental Health': [
            'Better outcomes with moderate to high diversity; extreme diversity may not be better',
            'Moderate engagement optimal; very high engagement may relate to higher symptoms',
            'Balanced versatility best; combination of genres, exploration, and language variety',
            'U-shaped relationship with briefer listening (0-1 hours) showing the lowest symptom scores'
        ]
    }
    
    summary_df = pd.DataFrame(optimal_summary)
    st.table(summary_df)
    
    # Final insights
    st.success("""
    **Key Takeaways for Implementation**
    
    The data shows consistent patterns pointing to moderation and balance in music consumption:
    
    1. **Goldilocks Principle**: There appears to be a "just right" amount for most music metrics, 
       with both too little and too much potentially relating to suboptimal mental health outcomes.
    
    2. **Personalized Approach**: Different mental health aspects (anxiety, depression, insomnia, OCD) 
       show slightly different optimal ranges, suggesting that personalized recommendations may be beneficial.
    
    3. **Musician Consideration**: Musicians, especially those both playing and composing, show higher 
       mental health symptom scores, suggesting they may benefit from targeted support or intervention programs.
    
    These evidence-based findings directly inform our KPIs and recommendations in the next section.
    """)

elif page == "KPIs & Recommendations":
    st.title("📈 KPIs & Recommendations")
    
    st.markdown("""
    This section synthesizes our analysis into actionable insights and key performance indicators (KPIs) 
    related to the relationship between music and mental health.
    """)
    
    # Create tabs for KPIs and Recommendations
    tabs = st.tabs(["Key Performance Indicators", "Strategic Recommendations"])
    
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
            optimal_range = "0-1 hours"
            st.markdown("""
            ### 📊 Optimal Listening Hours
            """)
            st.metric(
                label="Current Average",
                value=f"{avg_hours:.1f} hours/day",
                delta=f"Target range: {optimal_range}"
            )
            
            st.markdown("""
            **Why it matters**: Our analysis identified an optimal range of 0-1 hours of daily music listening associated with 
            lowest mental health symptom scores. This suggests that more limited, intentional music consumption may be 
            more beneficial than extended listening periods.
            
            **Target audience**: General population, particularly those using music as a coping mechanism for mental health.
            """)
            
            # Music Engagement Score
            avg_engagement = df_engineered['Engagement_Score'].mean()
            st.markdown("""
            ### 🎹 Music Engagement Score
            """)
            st.metric(
                label="Average Engagement Score",
                value=f"{avg_engagement:.1f}/10",
                delta="Moderate engagement may be optimal"
            )
            
            st.markdown("""
            **Why it matters**: Our analysis revealed that those with the highest engagement scores (particularly those who are both 
            instrumentalists and composers) reported higher mental health symptom levels. This suggests that highly engaged musicians 
            may be using music as a coping mechanism or experiencing industry-related stressors.
            
            **Target audience**: Music educators, mental health professionals, music therapy programs.
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
                delta="Medium to High diversity (4-8) optimal"
            )
            
            st.markdown("""
            **Why it matters**: Our ANOVA analysis identified the Medium to High diversity range (4-8) as having the lowest 
            mental health symptom scores. This suggests that exposure to a moderate variety of musical styles may have 
            beneficial effects, though the relationship is complex.
            
            **Target audience**: Streaming services, playlist curators, music recommendation systems.
            """)
            
            # Mental Health Awareness Score
            st.markdown("""
            ### 🧠 Mental Health Awareness Score
            """)
            st.metric(
                label="Optimal Mental Health Range",
                value="2-4 (Low symptoms)",
                delta="Current avg: 4.5"
            )
            
            st.markdown("""
            **Why it matters**: We identified a score of 2-4 on our mental health composite as the optimal range associated with 
            healthier music consumption patterns. This KPI helps identify when intervention or adjustment to music habits 
            might be beneficial.
            
            **Target audience**: Mental health professionals, individuals using music for self-care, wellness programs.
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
        
        # Summary of key findings
        st.info("""
        **Key Findings Summary**:
        
        Our analysis revealed several unexpected patterns in the data:
        
        1. People who are both instrumentalists and composers showed the highest mental health symptom scores, 
           suggesting that deeper musical engagement may be associated with higher reported mental health symptoms.
        
        2. The correlation matrix showed weak positive correlations between music variables and mental health symptoms,
           indicating that as music engagement increases, there may be a slight tendency toward higher symptom scores.
        
        3. An optimal listening time of 0-1 hours per day was associated with the lowest mental health symptom scores,
           with higher amounts showing increased symptom levels.
        
        4. Medium to high genre diversity (4-8 on our 10-point scale) showed the most favorable mental health outcomes.
        
        These findings highlight the complex, bidirectional relationship between music and mental health that
        requires careful interpretation and personalized approaches.
        """)
    
    # RECOMMENDATIONS TAB
    with tabs[1]:
        st.subheader("Strategic Recommendations")
        
        st.markdown("""
        Based on our comprehensive analysis of music listening habits and mental health, we've developed
        nuanced recommendations that reflect the complex relationship we observed:
        """)
        
        # Create two columns for recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎧 For Individuals
            
            **Balanced listening approach:**
            - Focus on brief, quality listening sessions (0-1 hours daily)
            - Explore different genres for a moderate diversity (4-8 range)
            - Be mindful of how music affects your mood and symptoms
            - If you're a musician experiencing symptoms, consider how music functions in your life
            
            **Why it works:**
            Our data shows that shorter, intentional music engagement (0-1 hours daily) is associated with the 
            lowest mental health symptom scores, suggesting that quality of listening may matter more than quantity.
            """)
            
            st.markdown("""
            ### 🏫 For Mental Health Professionals
            
            **Consider music in assessment:**
            - Assess how clients use music (coping, expression, occupation)
            - Be aware that musicians may have higher reported symptom levels
            - Explore the directions of influence between music and symptoms
            - Use moderate music engagement as a complementary intervention
            
            **Why it works:**
            Our analysis found that music creators often reported higher symptom levels, suggesting
            complex interactions between music creation and mental health.
            """)
        
        with col2:
            st.markdown("""
            ### 🎵 For Music Services
            
            **Evidence-based recommendation systems:**
            - Promote quality over quantity in music consumption
            - Encourage brief, focused listening sessions rather than extended play
            - Develop features to help users explore moderate genre diversity (4-8 range)
            - Provide mental health resources for musician users who may be struggling
            
            **Why it works:**
            Our analysis found that brief, intentional listening (0-1 hours daily) and moderate genre 
            diversity optimally support mental well-being, challenging the common industry goal of 
            maximizing engagement time.
            """)
            
            st.markdown("""
            ### 💼 For Music Educators & Industry
            
            **Support musician wellbeing:**
            - Recognize that musicians report higher mental health symptom levels
            - Integrate mental health awareness and coping strategies into music education
            - Create supportive environments that acknowledge the emotional intensity of music creation
            - Develop targeted mental health resources specifically for musicians
            
            **Why it works:**
            Our data consistently showed that those most engaged in music creation (both instrumentalists and composers)
            reported the highest mental health symptom levels, suggesting music creation may function as a coping 
            mechanism and/or the music industry may involve unique stressors requiring dedicated support.
            """)
        
        # Summary insight box
        st.info("""
        **Key Takeaway: Evidence-Based Insights**
        
        Our analysis revealed unexpected but consistent patterns:
        
        1. **Brief listening is beneficial**: 0-1 hours of daily music listening was consistently associated 
           with the lowest mental health symptom levels, challenging the common notion that more music is better.
           
        2. **Musicians report higher symptoms**: People engaged in music creation (especially those who both play 
           and compose) showed higher mental health symptom scores, suggesting music may function as a coping 
           mechanism rather than simply causing improved wellbeing.
           
        3. **Moderate genre diversity is optimal**: Medium to high genre diversity (4-8 range) showed better 
           mental health outcomes than either very narrow or extremely diverse listening habits.
           
        These findings highlight the complex, bidirectional relationship between music and mental health,
        challenging simplistic "music therapy" narratives and pointing to more nuanced approaches.
        """)


# Add a footer with information about the dashboard

