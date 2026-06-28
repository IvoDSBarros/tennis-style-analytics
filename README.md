# Elite Tennis Playing Styles from an AI Agent's Perspective

An interactive Plotly and CrewAI-powered application that analyzes tennis playing styles of new generation rising stars against the Big 3.

## 🌐 Live Application

**[https://IvoDSBarros.pythonanywhere.com](https://IvoDSBarros.pythonanywhere.com)**

*Optimized for desktop viewing*

## 📊 Data Pipeline

- **Source**: Tennis Match Charting Project (Jeff Sackmann)
- **Players**: 21 elite ATP players (2010-2024)
- **Matches**: 8 sampled matches per player per surface
- **Surfaces**: Clay, Grass, Hard
- **Metrics**: 14 continuous performance metrics optimized per surface via PCA

## 🧠 Architecture

### Pre-computation
- PCA performed per surface with optimized variable selection (see Methodology in app)
- Cluster analysis (Complete-Linkage, adaptive k selection)
- Results saved as JSON for instant access

### Runtime (Single Agent, One LLM Call)
- **Why single agent?** A multi-agent system introduces unnecessary complexity, reliability issues, and security risks. For this use case, a single agent with a well-crafted prompt and strict formatting rules is more robust and deterministic. PCA and clustering are purely statistical methods requiring no AI agent intervention.

1. **TennisDataEnforcer tool** retrieves:
   - Pre-calculated PCA scores, loadings, and variance
   - Natural clusters from PCA results
   - Player coordinates and cluster assignments

2. **Single LLM call** synthesizes data into:
   - **Scores narrative**: Player positioning, Sinner/Alcaraz vs Big 3 comparison, tactical clusters
   - **Loadings narrative**: What defines each PCA component in tennis terms

## 📁 Report Structure

| Section | Content |
|---------|---------|
| Scores | PCA scatter plot explanation, player positions, Sinner/Alcaraz vs Big 3 |
| Loadings | Variable contributions, tactical dimensions |

## 🛠️ Tech Stack

- **Frontend**: HTML/CSS/JavaScript, Plotly
- **Backend**: Python, CrewAI (single agent)
- **Data Processing**: Pandas, NumPy, Scikit-learn, SciPy
- **LLM**: Gemini 2.5 Flash

## 📂 Repository Structure

```
tennis-playing-styles/
├── ai_narrative/          # CrewAI-generated narratives
├── assets/                # Images and assets
├── data/                  # PCA scores, loadings, variance, clusters
├── preprocessing/         # PCA and clustering scripts
├── results/               # Analysis results
├── index.html             # Main application
├── styles.css             # Styling
├── script.js              # JavaScript logic
└── requirements.txt       # Dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Gemini API key

### Installation

```bash
# Clone repository
git clone https://github.com/IvoDSBarros/tennis-playing-styles.git
cd tennis-playing-styles

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Add your GEMINI_API_KEY to .env

# Run preprocessing (PCA + clustering)
python preprocessing/08_cluster_analysis_on_pca.py

# Generate AI narratives
python narrative_generator_tennis.py
```

## 📝 License

MIT