
import re
from transformers import BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_split_sentiments(user_input: str, aspects: str):
    # Split the input sentence by commas and periods, and remove extra spaces
    split_sentences = [sentence.strip() for sentence in re.split(r'[,./;:-]', user_input) if sentence.strip()]
    
        # Split the aspects by commas and remove extra spaces
    split_aspects = [aspect.strip() for aspect in aspects.split(",") if aspect.strip()]

    # Load pre-trained BERT tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Prepare result object
    results = []

    # Check if the number of sentences and aspects are the same
    if len(split_sentences) != len(split_aspects):
        print("Warning: Number of sentences and aspects do not match. Filtering out unmatched sentences.")

    # Analyze sentiment for each split sentence and aspect
    for sentence, aspect in zip(split_sentences, split_aspects):
        if not aspect:  # Skip if there is no aspect
            continue

        # Concatenate the sentence and aspect
        concatenated_sentence = f"{sentence} [SEP] {aspect}"

        # Tokenize and decode the sentence
        inputs = tokenizer(concatenated_sentence, return_tensors="pt", truncation=True, padding=True)
        text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        # Perform VADER sentiment analysis
        sentiment_scores = analyzer.polarity_scores(text)

        # Determine the overall sentiment
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Append the result for this segment
        results.append({
            "sentence": sentence,
            "aspect": aspect,
            #"sentiment_scores": sentiment_scores,
            "overall_sentiment": sentiment
        })

    return results

# Example usage
user_input = "The food was delicious, The service was slow / and the atmosphere was nice. because we went there at night"
aspects = "food, service, atmosphere, night"
result = analyze_split_sentiments(user_input, aspects)

# Print the results
for res in result:
    print(res)

