# Email Spam Classification System

## Overview

This project implements a comprehensive email spam classification system using machine learning techniques. The system can accurately classify emails as either spam or legitimate (ham) messages using various machine learning algorithms.

## Features

- **Multiple ML Algorithms**: Naive Bayes, Logistic Regression, Random Forest, and SVM
- **Advanced Text Preprocessing**: TF-IDF vectorization with n-gram features
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score metrics
- **Feature Analysis**: Identification of most important features for classification
- **Model Persistence**: Save and load trained models
- **Advanced Visualizations**: Interactive plots, statistical analysis, and data exploration
- **Word Cloud Analysis**: Visual representation of word frequencies
- **Interactive Dashboards**: Plotly-based interactive visualizations

## Visualization Features

### Static Visualizations
- **Data Distribution Analysis**: Text length, word count, and character analysis
- **Statistical Analysis**: Box plots, histograms, and correlation matrices
- **Word Frequency Analysis**: Top words in spam vs ham messages
- **Feature Importance**: Most important features for classification
- **Model Performance Comparison**: Side-by-side model evaluation

### Interactive Visualizations
- **Interactive Pie Charts**: Message type distribution
- **Interactive Scatter Plots**: Text length vs word count analysis
- **Interactive Box Plots**: Feature distributions by label
- **Interactive Histograms**: Text length distribution
- **Radar Charts**: Model performance comparison
- **Interactive Feature Importance**: Color-coded feature analysis

## Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script** (Windows):
   ```bash
   run_setup.bat
   ```
   
   Or manually:
   ```bash
   python setup.py
   ```

## Usage

### Quick Start

1. **Run the complete spam classification system**:
   ```bash
   python spam_classification.py
   ```

2. **Run the visualization dashboard**:
   ```bash
   python visualization_dashboard.py
   ```

3. **Test the classifier with sample texts**:
   ```bash
   python test_classifier.py
   ```

### Visualization Dashboard

The dedicated visualization dashboard provides comprehensive data analysis:

```bash
python visualization_dashboard.py
```

This will generate:
- **Static plots**: PNG files with detailed analysis
- **Interactive plots**: HTML files for web-based exploration
- **Statistical summaries**: Detailed numerical analysis
- **Word clouds**: Visual word frequency representation

### Generated Files

#### Static Visualizations
- `overview_dashboard.png` - Comprehensive data overview
- `statistical_analysis.png` - Statistical analysis by message type
- `correlation_matrix.png` - Feature correlation analysis
- `word_analysis.png` - Word frequency analysis
- `word_clouds.png` - Word cloud visualizations
- `feature_comparison.png` - Feature comparison charts
- `model_comparison.png` - Model performance comparison
- `confusion_matrix.png` - Confusion matrix for best model
- `feature_importance.png` - Feature importance analysis

#### Interactive Visualizations (HTML)
- `interactive_label_distribution.html` - Interactive pie chart
- `interactive_scatter_plot.html` - Interactive scatter plot
- `interactive_box_plots.html` - Interactive box plots
- `interactive_histogram.html` - Interactive histogram
- `interactive_model_comparison.html` - Interactive radar chart
- `interactive_model_bar_chart.html` - Interactive bar chart
- `interactive_feature_importance.html` - Interactive feature importance

## Project Structure

```
Week- (3)/
├── spam.csv                    # Dataset
├── spam_classification.py      # Main classification system
├── visualization_dashboard.py  # Dedicated visualization dashboard
├── test_classifier.py         # Test script
├── setup.py                   # Setup script
├── requirements.txt           # Dependencies
├── README.md                 # This file
└── run_setup.bat            # Windows setup script
```

## Data Analysis Features

### Text Analysis
- **Text Length Analysis**: Character and word count distributions
- **Character Analysis**: Uppercase, digit, and special character counts
- **Word Analysis**: Average word length and frequency analysis
- **Statistical Analysis**: Mean, standard deviation, min, max values

### Visualization Types
1. **Distribution Plots**: Histograms and density plots
2. **Comparison Plots**: Box plots and bar charts
3. **Correlation Analysis**: Heatmaps and scatter plots
4. **Word Analysis**: Frequency charts and word clouds
5. **Model Analysis**: Performance comparison and feature importance

## Key Insights

The visualization system helps identify:

- **Text Characteristics**: Spam messages tend to be longer with more uppercase characters
- **Word Patterns**: Different vocabulary usage between spam and ham
- **Feature Correlations**: Relationships between text features
- **Model Performance**: Which algorithms work best for this dataset
- **Feature Importance**: Which words/features are most predictive

## Advanced Features

### Interactive Exploration
- **Hover Information**: Detailed information on hover
- **Zoom and Pan**: Interactive navigation
- **Color Coding**: Visual distinction between categories
- **Multiple Views**: Different perspectives on the same data

### Statistical Analysis
- **Descriptive Statistics**: Comprehensive numerical summaries
- **Correlation Analysis**: Feature relationships
- **Distribution Analysis**: Shape and spread of data
- **Outlier Detection**: Identification of unusual patterns

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Encoding Issues**: The system automatically tries multiple encodings

3. **Memory Issues**: For large datasets, reduce `max_features` in `extract_features()`

4. **WordCloud Issues**: Install wordcloud if needed
   ```bash
   pip install wordcloud
   ```

### Performance Tips

- Use smaller `max_features` for faster processing
- Reduce dataset size for quick testing
- Use saved models for repeated predictions
- Close matplotlib windows to free memory

## Contributing

Feel free to enhance the visualization features by:
- Adding new plot types
- Improving interactive features
- Optimizing performance
- Adding new statistical analyses

## License

This project is for educational purposes as part of the Data Science Internship program.

---

**Note**: The visualization system requires additional dependencies (plotly, wordcloud) which are included in the requirements.txt file. Make sure to install all dependencies for the full visualization experience. 