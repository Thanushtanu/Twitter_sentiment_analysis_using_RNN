# Twitter Sentiment Analysis Using RNN

## Overview

This project implements a Recurrent Neural Network (RNN) to perform sentiment analysis on tweets. The goal is to classify tweets as positive, negative, or neutral based on their content. This can be useful for understanding public sentiment on various topics, brands, or events.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Collects tweets using Twitter API.
- Preprocesses text data (tokenization, removing stop words, etc.).
- Implements RNN architecture for sentiment classification.
- Visualizes training and validation results.
- Provides an interactive interface for real-time sentiment analysis.

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Twitter API (Tweepy)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Thanushtanu/Twitter_sentiment_analysis_using_RNN.git

2. Navigate to the project directory:
   ```bash
   cd Twitter_sentiment_analysis_using_RNN
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set up your Twitter API credentials in the `config.py` file.
2. Run the data collection script:
   ```bash
   python collect_tweets.py
   ```
3. Train the RNN model:
   ```bash
   python train_model.py
   ```
4. To analyze a new tweet, run:
   ```bash
   python analyze_tweet.py "Your tweet here"
   ```

## Data

The dataset used for training the model consists of labeled tweets collected from Twitter. The data is structured in a CSV format with columns for the tweet text and its corresponding sentiment label.

## Model Architecture

The RNN model is built using LSTM (Long Short-Term Memory) units to capture temporal dependencies in the tweet data. The architecture includes:

- Embedding Layer
- LSTM Layer(s)
- Fully Connected Output Layer

## Results

The model's performance can be evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualization of training and validation loss is also provided.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
